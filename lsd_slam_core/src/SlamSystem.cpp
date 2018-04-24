/**
* This file is part of LSD-SLAM.
*
* Copyright 2013 Jakob Engel <engelj at in dot tum dot de> (Technical University of Munich)
* For more information see <http://vision.in.tum.de/lsdslam> 
*
* LSD-SLAM is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* LSD-SLAM is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with LSD-SLAM. If not, see <http://www.gnu.org/licenses/>.
*/

#include "SlamSystem.h"

#include "DataStructures/Frame.h"
#include "Tracking/SE3Tracker.h"
#include "Tracking/Sim3Tracker.h"
#include "DepthEstimation/DepthMap.h"
#include "Tracking/TrackingReference.h"
#include "LiveSLAMWrapper.h"
#include "util/globalFuncs.h"
#include "GlobalMapping/KeyFrameGraph.h"
#include "GlobalMapping/TrackableKeyFrameSearch.h"
#include "GlobalMapping/g2oTypeSim3Sophus.h"
#include "IOWrapper/ImageDisplay.h"
#include "IOWrapper/Output3DWrapper.h"
#include <g2o/core/robust_kernel_impl.h>
#include "DataStructures/FrameMemory.h"
#include "deque"

// for mkdir
#include <sys/types.h>
#include <sys/stat.h>

#ifdef ANDROID
#include <android/log.h>
#endif

#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "DepthEstimation/DepthMapPixelHypothesis.h"

using namespace lsd_slam;


SlamSystem::SlamSystem(int w, int h, Eigen::Matrix3f K, bool enableSLAM)
: SLAMEnabled(enableSLAM), relocalizer(w,h,K)
{
	if(w%16 != 0 || h%16!=0)
	{
		printf("image dimensions must be multiples of 16! Please crop your images / video accordingly.\n");
		assert(false);
	}

	this->width = w;
	this->height = h;
	this->K = K;
	trackingIsGood = true;


	currentKeyFrame =  nullptr;
	trackingReferenceFrameSharedPT = nullptr;
	keyFrameGraph = new KeyFrameGraph();
	createNewKeyFrame = false;

	map =  new DepthMap(w,h,K);
	
	newConstraintAdded = false;
	haveUnmergedOptimizationOffset = false;


	tracker = new SE3Tracker(w,h,K);
	// Do not use more than 4 levels for odometry tracking
	for (int level = 4; level < PYRAMID_LEVELS; ++level)
		tracker->settings.maxItsPerLvl[level] = 0;
	trackingReference = new TrackingReference();
	mappingTrackingReference = new TrackingReference();


	if(SLAMEnabled)
	{
		trackableKeyFrameSearch = new TrackableKeyFrameSearch(keyFrameGraph,w,h,K);
		constraintTracker = new Sim3Tracker(w,h,K);
		constraintSE3Tracker = new SE3Tracker(w,h,K);
		newKFTrackingReference = new TrackingReference();
		candidateTrackingReference = new TrackingReference();
	}
	else
	{
		constraintSE3Tracker = 0;
		trackableKeyFrameSearch = 0;
		constraintTracker = 0;
		newKFTrackingReference = 0;
		candidateTrackingReference = 0;
	}


	outputWrapper = 0;

	keepRunning = true;
	doFinalOptimization = false;
	depthMapScreenshotFlag = false;
	lastTrackingClosenessScore = 0;

	thread_mapping = boost::thread(&SlamSystem::mappingThreadLoop, this);

	if(SLAMEnabled)
	{
		thread_constraint_search = boost::thread(&SlamSystem::constraintSearchThreadLoop, this);
		thread_optimization = boost::thread(&SlamSystem::optimizationThreadLoop, this);
	}



	msTrackFrame = msOptimizationIteration = msFindConstraintsItaration = msFindReferences = 0;
	nTrackFrame = nOptimizationIteration = nFindConstraintsItaration = nFindReferences = 0;
	nAvgTrackFrame = nAvgOptimizationIteration = nAvgFindConstraintsItaration = nAvgFindReferences = 0;
	gettimeofday(&lastHzUpdate, NULL);
        
        curDepth = new float[w*h];
        curImage ;//= new float[w*h];
        
        correctedDepth = new float[w*h];
        prevCorrectedDepth = new float[w*h];
        prevCorrectVar = new float[w*h];
        correctedDepthVar = new float[w*h];
        
         prevDepthCorrectionAvilable = false;
         
         scalesum = 0;
         runCount = 0;
         fuseAllframes = true;
        sensorDepth_ = NULL;
}

SlamSystem::~SlamSystem()
{
	keepRunning = false;

	// make sure none is waiting for something.
	printf("... waiting for SlamSystem's threads to exit\n");
	newFrameMappedSignal.notify_all();
	unmappedTrackedFramesSignal.notify_all();
	newKeyFrameCreatedSignal.notify_all();
	newConstraintCreatedSignal.notify_all();

	thread_mapping.join();
	thread_constraint_search.join();
	thread_optimization.join();
	printf("DONE waiting for SlamSystem's threads to exit\n");

	if(trackableKeyFrameSearch != 0) delete trackableKeyFrameSearch;
	if(constraintTracker != 0) delete constraintTracker;
	if(constraintSE3Tracker != 0) delete constraintSE3Tracker;
	if(newKFTrackingReference != 0) delete newKFTrackingReference;
	if(candidateTrackingReference != 0) delete candidateTrackingReference;

	delete mappingTrackingReference;
	delete map;
	delete trackingReference;
	delete tracker;

	// make shure to reset all shared pointers to all frames before deleting the keyframegraph!
	unmappedTrackedFrames.clear();
	latestFrameTriedForReloc.reset();
	latestTrackedFrame.reset();
	currentKeyFrame.reset();
	trackingReferenceFrameSharedPT.reset();

	// delte keyframe graph
	delete keyFrameGraph;

	FrameMemory::getInstance().releaseBuffes();


	Util::closeAllWindows();
        delete[] curDepth;
        
}

void SlamSystem::setVisualization(Output3DWrapper* outputWrapper)
{
	this->outputWrapper = outputWrapper;
}

void SlamSystem::mergeOptimizationOffset()
{
	// update all vertices that are in the graph!
	poseConsistencyMutex.lock();

	bool needPublish = false;
	if(haveUnmergedOptimizationOffset)
	{
		keyFrameGraph->keyframesAllMutex.lock_shared();
		for(unsigned int i=0;i<keyFrameGraph->keyframesAll.size(); i++)
			keyFrameGraph->keyframesAll[i]->pose->applyPoseGraphOptResult();
		keyFrameGraph->keyframesAllMutex.unlock_shared();

		haveUnmergedOptimizationOffset = false;
		needPublish = true;
	}

	poseConsistencyMutex.unlock();






	if(needPublish)
		publishKeyframeGraph();
}



void SlamSystem::mappingThreadLoop()
{
	printf("Started mapping thread!\n");
	while(keepRunning)
	{
		if (!doMappingIteration())
		{
			boost::unique_lock<boost::mutex> lock(unmappedTrackedFramesMutex);
			unmappedTrackedFramesSignal.timed_wait(lock,boost::posix_time::milliseconds(200));	// slight chance of deadlock otherwise
			lock.unlock();
		}

		newFrameMappedMutex.lock();
		newFrameMappedSignal.notify_all();
		newFrameMappedMutex.unlock();
	}
	printf("Exited mapping thread \n");
}

void SlamSystem::finalize()
{
	printf("Finalizing Graph... finding final constraints!!\n");

	lastNumConstraintsAddedOnFullRetrack = 1;
	while(lastNumConstraintsAddedOnFullRetrack != 0)
	{
		doFullReConstraintTrack = true;
		usleep(200000);
	}


	printf("Finalizing Graph... optimizing!!\n");
	doFinalOptimization = true;
	newConstraintMutex.lock();
	newConstraintAdded = true;
	newConstraintCreatedSignal.notify_all();
	newConstraintMutex.unlock();
	while(doFinalOptimization)
	{
		usleep(200000);
	}


	printf("Finalizing Graph... publishing!!\n");
	unmappedTrackedFramesMutex.lock();
	unmappedTrackedFramesSignal.notify_one();
	unmappedTrackedFramesMutex.unlock();
	while(doFinalOptimization)
	{
		usleep(200000);
	}
	boost::unique_lock<boost::mutex> lock(newFrameMappedMutex);
	newFrameMappedSignal.wait(lock);
	newFrameMappedSignal.wait(lock);

	usleep(200000);
	printf("Done Finalizing Graph.!!\n");
}


void SlamSystem::constraintSearchThreadLoop()
{
	printf("Started  constraint search thread!\n");
	
	boost::unique_lock<boost::mutex> lock(newKeyFrameMutex);
	int failedToRetrack = 0;

	while(keepRunning)
	{
		if(newKeyFrames.size() == 0)
		{
			lock.unlock();
			keyFrameGraph->keyframesForRetrackMutex.lock();
			bool doneSomething = false;
			if(keyFrameGraph->keyframesForRetrack.size() > 10)
			{
				std::deque< Frame* >::iterator toReTrack = keyFrameGraph->keyframesForRetrack.begin() + (rand() % (keyFrameGraph->keyframesForRetrack.size()/3));
				Frame* toReTrackFrame = *toReTrack;

				keyFrameGraph->keyframesForRetrack.erase(toReTrack);
				keyFrameGraph->keyframesForRetrack.push_back(toReTrackFrame);

				keyFrameGraph->keyframesForRetrackMutex.unlock();

				int found = findConstraintsForNewKeyFrames(toReTrackFrame, false, false, 2.0);
				if(found == 0)
					failedToRetrack++;
				else
					failedToRetrack=0;

				if(failedToRetrack < (int)keyFrameGraph->keyframesForRetrack.size() - 5)
					doneSomething = true;
			}
			else
				keyFrameGraph->keyframesForRetrackMutex.unlock();

			lock.lock();

			if(!doneSomething)
			{
				if(enablePrintDebugInfo && printConstraintSearchInfo)
					printf("nothing to re-track... waiting.\n");
				newKeyFrameCreatedSignal.timed_wait(lock,boost::posix_time::milliseconds(500));

			}
		}
		else
		{
			Frame* newKF = newKeyFrames.front();
			newKeyFrames.pop_front();
			lock.unlock();

			struct timeval tv_start, tv_end;
			gettimeofday(&tv_start, NULL);

			findConstraintsForNewKeyFrames(newKF, true, true, 1.0);
			failedToRetrack=0;
			gettimeofday(&tv_end, NULL);
			msFindConstraintsItaration = 0.9*msFindConstraintsItaration + 0.1*((tv_end.tv_sec-tv_start.tv_sec)*1000.0f + (tv_end.tv_usec-tv_start.tv_usec)/1000.0f);
			nFindConstraintsItaration++;

			FrameMemory::getInstance().pruneActiveFrames();
			lock.lock();
		}


		if(doFullReConstraintTrack)
		{
			lock.unlock();
			printf("Optizing Full Map!\n");

			int added = 0;
			for(unsigned int i=0;i<keyFrameGraph->keyframesAll.size();i++)
			{
				if(keyFrameGraph->keyframesAll[i]->pose->isInGraph)
					added += findConstraintsForNewKeyFrames(keyFrameGraph->keyframesAll[i], false, false, 1.0);
			}

			printf("Done optizing Full Map! Added %d constraints.\n", added);

			doFullReConstraintTrack = false;

			lastNumConstraintsAddedOnFullRetrack = added;
			lock.lock();
		}



	}

	printf("Exited constraint search thread \n");
}

void SlamSystem::optimizationThreadLoop()
{
	printf("Started optimization thread \n");

	while(keepRunning)
	{
		boost::unique_lock<boost::mutex> lock(newConstraintMutex);
		if(!newConstraintAdded)
			newConstraintCreatedSignal.timed_wait(lock,boost::posix_time::milliseconds(2000));	// slight chance of deadlock otherwise
		newConstraintAdded = false;
		lock.unlock();

		if(doFinalOptimization)
		{
			printf("doing final optimization iteration!\n");
			optimizationIteration(50, 0.001);
			doFinalOptimization = false;
		}
		while(optimizationIteration(5, 0.02));
	}

	printf("Exited optimization thread \n");
}

void SlamSystem::publishKeyframeGraph()
{
	if (outputWrapper != nullptr)
		outputWrapper->publishKeyframeGraph(keyFrameGraph);
}

void SlamSystem::requestDepthMapScreenshot(const std::string& filename)
{
	depthMapScreenshotFilename = filename;
	depthMapScreenshotFlag = true;
}

void SlamSystem::finishCurrentKeyframe()
{
	if(enablePrintDebugInfo && printThreadingInfo)
		printf("FINALIZING KF %d\n", currentKeyFrame->id());

	map->finalizeKeyFrame();

	if(SLAMEnabled)
	{
		mappingTrackingReference->importFrame(currentKeyFrame.get());
		currentKeyFrame->setPermaRef(mappingTrackingReference);
		mappingTrackingReference->invalidate();

		if(currentKeyFrame->idxInKeyframes < 0)
		{
			keyFrameGraph->keyframesAllMutex.lock();
			currentKeyFrame->idxInKeyframes = keyFrameGraph->keyframesAll.size();
			keyFrameGraph->keyframesAll.push_back(currentKeyFrame.get());
			keyFrameGraph->totalPoints += currentKeyFrame->numPoints;
			keyFrameGraph->totalVertices ++;
			keyFrameGraph->keyframesAllMutex.unlock();

			newKeyFrameMutex.lock();
			newKeyFrames.push_back(currentKeyFrame.get());
			newKeyFrameCreatedSignal.notify_all();
			newKeyFrameMutex.unlock();
		}
	}
        
        //updateGlobalScale();
	//if(outputWrapper!= 0)
//		outputWrapper->publishKeyframe(currentKeyFrame.get(),sensorDepth_,curDepth,curImage,currentFrame,correctedDepth,prevCorrectedDepth);
}

void SlamSystem::discardCurrentKeyframe()
{
	if(enablePrintDebugInfo && printThreadingInfo)
		printf("DISCARDING KF %d\n", currentKeyFrame->id());

	if(currentKeyFrame->idxInKeyframes >= 0)
	{
		printf("WARNING: trying to discard a KF that has already been added to the graph... finalizing instead.\n");
		finishCurrentKeyframe();
		return;
	}


	map->invalidate();

	keyFrameGraph->allFramePosesMutex.lock();
	for(FramePoseStruct* p : keyFrameGraph->allFramePoses)
	{
		if(p->trackingParent != 0 && p->trackingParent->frameID == currentKeyFrame->id())
			p->trackingParent = 0;
	}
	keyFrameGraph->allFramePosesMutex.unlock();


	keyFrameGraph->idToKeyFrameMutex.lock();
	keyFrameGraph->idToKeyFrame.erase(currentKeyFrame->id());
	keyFrameGraph->idToKeyFrameMutex.unlock();

}

void SlamSystem::createNewCurrentKeyframe(std::shared_ptr<Frame> newKeyframeCandidate)
{
	if(enablePrintDebugInfo && printThreadingInfo)
		printf("CREATE NEW KF %d from %d\n", newKeyframeCandidate->id(), currentKeyFrame->id());


	if(SLAMEnabled)
	{
		// add NEW keyframe to id-lookup
		keyFrameGraph->idToKeyFrameMutex.lock();
		keyFrameGraph->idToKeyFrame.insert(std::make_pair(newKeyframeCandidate->id(), newKeyframeCandidate));
		keyFrameGraph->idToKeyFrameMutex.unlock();
	}

	// propagate & make new.
	map->createKeyFrame(newKeyframeCandidate.get());

	if(printPropagationStatistics)
	{

		Eigen::Matrix<float, 20, 1> data;
		data.setZero();
		data[0] = runningStats.num_prop_attempts / ((float)width*height);
		data[1] = (runningStats.num_prop_created + runningStats.num_prop_merged) / (float)runningStats.num_prop_attempts;
		data[2] = runningStats.num_prop_removed_colorDiff / (float)runningStats.num_prop_attempts;

		outputWrapper->publishDebugInfo(data);
	}

	currentKeyFrameMutex.lock();
	currentKeyFrame = newKeyframeCandidate;
        //updateGlobalScale();
	currentKeyFrameMutex.unlock();
        
        
         if(outputWrapper!= 0)
	{	
            outputWrapper->publishKeyframe(currentKeyFrame.get(),(float*)currentKeyFrame.get()->sensordepth.data,curDepth,curImage,currentFrame,correctedDepth,prevCorrectedDepth,currentKeyFrame.get());
        }
}
void SlamSystem::loadNewCurrentKeyframe(Frame* keyframeToLoad)
{
	if(enablePrintDebugInfo && printThreadingInfo)
		printf("RE-ACTIVATE KF %d\n", keyframeToLoad->id());

	map->setFromExistingKF(keyframeToLoad);

	if(enablePrintDebugInfo && printRegularizeStatistics)
		printf("re-activate frame %d!\n", keyframeToLoad->id());

	currentKeyFrameMutex.lock();
	currentKeyFrame = keyFrameGraph->idToKeyFrame.find(keyframeToLoad->id())->second;
	currentKeyFrame->depthHasBeenUpdatedFlag = false;
        //updateGlobalScale();
	currentKeyFrameMutex.unlock();
}

void SlamSystem::changeKeyframe(bool noCreate, bool force, float maxScore)
{
	Frame* newReferenceKF=0;
	std::shared_ptr<Frame> newKeyframeCandidate = latestTrackedFrame;
	if(doKFReActivation && SLAMEnabled)
	{
		struct timeval tv_start, tv_end;
		gettimeofday(&tv_start, NULL);
		newReferenceKF = trackableKeyFrameSearch->findRePositionCandidate(newKeyframeCandidate.get(), maxScore);
		gettimeofday(&tv_end, NULL);
		msFindReferences = 0.9*msFindReferences + 0.1*((tv_end.tv_sec-tv_start.tv_sec)*1000.0f + (tv_end.tv_usec-tv_start.tv_usec)/1000.0f);
		nFindReferences++;
	}

	if(newReferenceKF != 0)
		loadNewCurrentKeyframe(newReferenceKF);
	else
	{
		if(force)
		{
			if(noCreate)
			{
				trackingIsGood = false;
				nextRelocIdx = -1;
				printf("mapping is disabled & moved outside of known map. Starting Relocalizer!\n");
			}
			else
				createNewCurrentKeyframe(newKeyframeCandidate);
		}
	}


	createNewKeyFrame = false;
}

bool SlamSystem::updateKeyframe()
{
	std::shared_ptr<Frame> reference = nullptr;
	std::deque< std::shared_ptr<Frame> > references;

	unmappedTrackedFramesMutex.lock();

	// remove frames that have a different tracking parent.
	while(unmappedTrackedFrames.size() > 0 &&
			(!unmappedTrackedFrames.front()->hasTrackingParent() ||
					unmappedTrackedFrames.front()->getTrackingParent() != currentKeyFrame.get()))
	{
		unmappedTrackedFrames.front()->clear_refPixelWasGood();
		unmappedTrackedFrames.pop_front();
	}

	// clone list
	if(unmappedTrackedFrames.size() > 0)
	{
		for(unsigned int i=0;i<unmappedTrackedFrames.size(); i++)
			references.push_back(unmappedTrackedFrames[i]);

		std::shared_ptr<Frame> popped = unmappedTrackedFrames.front();
		unmappedTrackedFrames.pop_front();
		unmappedTrackedFramesMutex.unlock();

		if(enablePrintDebugInfo && printThreadingInfo)
			printf("MAPPING %d on %d to %d (%d frames)\n", currentKeyFrame->id(), references.front()->id(), references.back()->id(), (int)references.size());

		map->updateKeyframe(references);

		popped->clear_refPixelWasGood();
		references.clear();
	}
	else
	{
		unmappedTrackedFramesMutex.unlock();
		return false;
	}


	if(enablePrintDebugInfo && printRegularizeStatistics)
	{
		Eigen::Matrix<float, 20, 1> data;
		data.setZero();
		data[0] = runningStats.num_reg_created;
		data[2] = runningStats.num_reg_smeared;
		data[3] = runningStats.num_reg_deleted_secondary;
		data[4] = runningStats.num_reg_deleted_occluded;
		data[5] = runningStats.num_reg_blacklisted;

		data[6] = runningStats.num_observe_created;
		data[7] = runningStats.num_observe_create_attempted;
		data[8] = runningStats.num_observe_updated;
		data[9] = runningStats.num_observe_update_attempted;


		data[10] = runningStats.num_observe_good;
		data[11] = runningStats.num_observe_inconsistent;
		data[12] = runningStats.num_observe_notfound;
		data[13] = runningStats.num_observe_skip_oob;
		data[14] = runningStats.num_observe_skip_fail;

		outputWrapper->publishDebugInfo(data);
	}


        //updateGlobalScale();
	//if(outputWrapper != 0 && continuousPCOutput && currentKeyFrame != 0)
	//	outputWrapper->publishKeyframe(currentKeyFrame.get(),sensorDepth_,curDepth,curImage,currentFrame,correctedDepth,prevCorrectedDepth);

	return true;
}

void SlamSystem::updateGlobalScale()
{
    int level = 0;
    Frame* kFrame = currentKeyFrame.get();
    int w = kFrame->width(level);
    int h = kFrame->height(level);
    const float* invDepht = kFrame->idepth(level);
    float lsdSum = 0;
    float sensorSum = 0;
    for(int i =0; i < w * h ; i++ ){
        float kfInvDep = invDepht[i];
        float sensorDep = sensorDepth_[i];
        if((!std::isnan(kfInvDep)  & kfInvDep  >0) & (!std::isnan(sensorDep)  & sensorDep  >0)){
            lsdSum += 1/kfInvDep;
            sensorSum += sensorDep;
       
        }
      
    }
      globalScale = sensorSum / lsdSum;
      //std::cout <<"scale "<<globalScale<<std::endl;
      
      
       if(!fuseAllframes & sensorDepth_ != NULL)
        {
           SE3 newRefToFrame_poseUpdate;
        //getCurentDepthMap( currentKeyFrame.get(), newRefToFrame_poseUpdate,  trackingReference,sensorDepth_);
                
        
        //if(outputWrapper!= 0)
		//outputWrapper->publishKeyframe(currentKeyFrame.get(),sensorDepth_,curDepth,curImage,currentFrame,correctedDepth,prevCorrectedDepth);

        }
        
	
}
void SlamSystem::addTimingSamples()
{
	map->addTimingSample();
	struct timeval now;
	gettimeofday(&now, NULL);
	float sPassed = ((now.tv_sec-lastHzUpdate.tv_sec) + (now.tv_usec-lastHzUpdate.tv_usec)/1000000.0f);
	if(sPassed > 1.0f)
	{
		nAvgTrackFrame = 0.8*nAvgTrackFrame + 0.2*(nTrackFrame / sPassed); nTrackFrame = 0;
		nAvgOptimizationIteration = 0.8*nAvgOptimizationIteration + 0.2*(nOptimizationIteration / sPassed); nOptimizationIteration = 0;
		nAvgFindReferences = 0.8*nAvgFindReferences + 0.2*(nFindReferences / sPassed); nFindReferences = 0;

		if(trackableKeyFrameSearch != 0)
		{
			trackableKeyFrameSearch->nAvgTrackPermaRef = 0.8*trackableKeyFrameSearch->nAvgTrackPermaRef + 0.2*(trackableKeyFrameSearch->nTrackPermaRef / sPassed); trackableKeyFrameSearch->nTrackPermaRef = 0;
		}
		nAvgFindConstraintsItaration = 0.8*nAvgFindConstraintsItaration + 0.2*(nFindConstraintsItaration / sPassed); nFindConstraintsItaration = 0;
		nAvgOptimizationIteration = 0.8*nAvgOptimizationIteration + 0.2*(nOptimizationIteration / sPassed); nOptimizationIteration = 0;

		lastHzUpdate = now;


		if(enablePrintDebugInfo && printOverallTiming)
		{
			printf("MapIt: %3.1fms (%.1fHz); Track: %3.1fms (%.1fHz); Create: %3.1fms (%.1fHz); FindRef: %3.1fms (%.1fHz); PermaTrk: %3.1fms (%.1fHz); Opt: %3.1fms (%.1fHz); FindConst: %3.1fms (%.1fHz);\n",
					map->msUpdate, map->nAvgUpdate,
					msTrackFrame, nAvgTrackFrame,
					map->msCreate+map->msFinalize, map->nAvgCreate,
					msFindReferences, nAvgFindReferences,
					trackableKeyFrameSearch != 0 ? trackableKeyFrameSearch->msTrackPermaRef : 0, trackableKeyFrameSearch != 0 ? trackableKeyFrameSearch->nAvgTrackPermaRef : 0,
					msOptimizationIteration, nAvgOptimizationIteration,
					msFindConstraintsItaration, nAvgFindConstraintsItaration);
		}
	}

}


void SlamSystem::debugDisplayDepthMap()
{


	map->debugPlotDepthMap();
	double scale = 1;
	if(currentKeyFrame != 0 && currentKeyFrame != 0)
		scale = currentKeyFrame->getScaledCamToWorld().scale();
	// debug plot depthmap
	char buf1[200];
	char buf2[200];


	snprintf(buf1,200,"Map: Upd %3.0fms (%2.0fHz); Trk %3.0fms (%2.0fHz); %d / %d / %d",
			map->msUpdate, map->nAvgUpdate,
			msTrackFrame, nAvgTrackFrame,
			currentKeyFrame->numFramesTrackedOnThis, currentKeyFrame->numMappedOnThis, (int)unmappedTrackedFrames.size());

	snprintf(buf2,200,"dens %2.0f%%; good %2.0f%%; scale %2.2f; res %2.1f/; usg %2.0f%%; Map: %d F, %d KF, %d E, %.1fm Pts",
			100*currentKeyFrame->numPoints/(float)(width*height),
			100*tracking_lastGoodPerBad,
			scale,
			tracking_lastResidual,
			100*tracking_lastUsage,
			(int)keyFrameGraph->allFramePoses.size(),
			keyFrameGraph->totalVertices,
			(int)keyFrameGraph->edgesAll.size(),
			1e-6 * (float)keyFrameGraph->totalPoints);


	if(onSceenInfoDisplay)
		printMessageOnCVImage(map->debugImageDepth, buf1, buf2);
	if (displayDepthMap)
		Util::displayImage( "DebugWindow DEPTH", map->debugImageDepth, false );

	int pressedKey = Util::waitKey(1);
	handleKey(pressedKey);
}


void SlamSystem::takeRelocalizeResult()
{
	Frame* keyframe;
	int succFrameID;
	SE3 succFrameToKF_init;
	std::shared_ptr<Frame> succFrame;
	relocalizer.stop();
	relocalizer.getResult(keyframe, succFrame, succFrameID, succFrameToKF_init);
	assert(keyframe != 0);

	loadNewCurrentKeyframe(keyframe);

	currentKeyFrameMutex.lock();
	trackingReference->importFrame(currentKeyFrame.get());
	trackingReferenceFrameSharedPT = currentKeyFrame;
	currentKeyFrameMutex.unlock();

	tracker->trackFrame(
			trackingReference,
			succFrame.get(),
			succFrameToKF_init);

	if(!tracker->trackingWasGood || tracker->lastGoodCount / (tracker->lastGoodCount + tracker->lastBadCount) < 1-0.75f*(1-MIN_GOODPERGOODBAD_PIXEL))
	{
		if(enablePrintDebugInfo && printRelocalizationInfo)
			printf("RELOCALIZATION FAILED BADLY! discarding result.\n");
		trackingReference->invalidate();
	}
	else
	{
		keyFrameGraph->addFrame(succFrame.get());

		unmappedTrackedFramesMutex.lock();
		if(unmappedTrackedFrames.size() < 50)
			unmappedTrackedFrames.push_back(succFrame);
		unmappedTrackedFramesMutex.unlock();

		currentKeyFrameMutex.lock();
		createNewKeyFrame = false;
		trackingIsGood = true;
		currentKeyFrameMutex.unlock();
	}
}

bool SlamSystem::doMappingIteration()
{
	if(currentKeyFrame == 0)
		return false;

	if(!doMapping && currentKeyFrame->idxInKeyframes < 0)
	{
		if(currentKeyFrame->numMappedOnThisTotal >= MIN_NUM_MAPPED)
			finishCurrentKeyframe();
		else
			discardCurrentKeyframe();

		map->invalidate();
		printf("Finished KF %d as Mapping got disabled!\n",currentKeyFrame->id());

		changeKeyframe(true, true, 1.0f);
	}

	mergeOptimizationOffset();
	addTimingSamples();

	if(dumpMap)
	{
		keyFrameGraph->dumpMap(packagePath+"/save");
		dumpMap = false;
	}


	// set mappingFrame
	if(trackingIsGood)
	{
		if(!doMapping)
		{
			//printf("tryToChange refframe, lastScore %f!\n", lastTrackingClosenessScore);
			if(lastTrackingClosenessScore > 1)
				changeKeyframe(true, false, lastTrackingClosenessScore * 0.75);

			if (displayDepthMap || depthMapScreenshotFlag)
				debugDisplayDepthMap();

			return false;
		}


		if (createNewKeyFrame)
		{
			finishCurrentKeyframe();
			changeKeyframe(false, true, 1.0f);


			if (displayDepthMap || depthMapScreenshotFlag)
				debugDisplayDepthMap();
		}
		else
		{
			bool didSomething = updateKeyframe();

			if (displayDepthMap || depthMapScreenshotFlag)
				debugDisplayDepthMap();
			if(!didSomething)
				return false;
		}

		return true;
	}
	else
	{
		// invalidate map if it was valid.
		if(map->isValid())
		{
			if(currentKeyFrame->numMappedOnThisTotal >= MIN_NUM_MAPPED)
				finishCurrentKeyframe();
			else
				discardCurrentKeyframe();

			map->invalidate();
		}

		// start relocalizer if it isnt running already
		if(!relocalizer.isRunning)
			relocalizer.start(keyFrameGraph->keyframesAll);

		// did we find a frame to relocalize with?
		if(relocalizer.waitResult(50))
			takeRelocalizeResult();


		return true;
	}
}


void SlamSystem::gtDepthInit(uchar* image, const cv::Mat& depth, double timeStamp, int id)
{
	printf("Doing GT initialization!\n");

	currentKeyFrameMutex.lock();
        
	currentKeyFrame.reset(new Frame(id, width, height, K, timeStamp, image,depth));
	currentKeyFrame->setDepthFromGroundTruth((float*)depth.data);
        globalScale = 1;
	map->initializeFromGTDepth(currentKeyFrame.get());
	keyFrameGraph->addFrame(currentKeyFrame.get());

	currentKeyFrameMutex.unlock();

	if(doSlam)
	{
		keyFrameGraph->idToKeyFrameMutex.lock();
		keyFrameGraph->idToKeyFrame.insert(std::make_pair(currentKeyFrame->id(), currentKeyFrame));
		keyFrameGraph->idToKeyFrameMutex.unlock();
	}
	if(continuousPCOutput && outputWrapper != 0) outputWrapper->publishKeyframe(currentKeyFrame.get(),sensorDepth_,curDepth,curImage,currentFrame,correctedDepth,prevCorrectedDepth,currentKeyFrame.get());

	printf("Done GT initialization!\n");
}


void SlamSystem::randomInit(uchar* image, double timeStamp, int id)
{
	printf("Doing Random initialization!\n");

	if(!doMapping)
		printf("WARNING: mapping is disabled, but we just initialized... THIS WILL NOT WORK! Set doMapping to true.\n");


	currentKeyFrameMutex.lock();
        cv::Mat dummy;
	currentKeyFrame.reset(new Frame(id, width, height, K, timeStamp, image,dummy));
        globalScale = 1;
	map->initializeRandomly(currentKeyFrame.get());
	keyFrameGraph->addFrame(currentKeyFrame.get());

	currentKeyFrameMutex.unlock();

	if(doSlam)
	{
		keyFrameGraph->idToKeyFrameMutex.lock();
		keyFrameGraph->idToKeyFrame.insert(std::make_pair(currentKeyFrame->id(), currentKeyFrame));
		keyFrameGraph->idToKeyFrameMutex.unlock();
	}
	if(continuousPCOutput && outputWrapper != 0) outputWrapper->publishKeyframe(currentKeyFrame.get(),sensorDepth_,curDepth,curImage,currentFrame,correctedDepth,prevCorrectedDepth,currentKeyFrame.get());


	if (displayDepthMap || depthMapScreenshotFlag)
		debugDisplayDepthMap();


	printf("Done Random initialization!\n");

}

void SlamSystem::trackFrame(uchar* image, unsigned int frameID, bool blockUntilMapped, double timestamp , const cv::Mat& sensorDepth)
{
    
        sensorDepth_ = (float*)sensorDepth.data;
	// Create new frame
	std::shared_ptr<Frame> trackingNewFrame(new Frame(frameID, width, height, K, timestamp, image,sensorDepth));

	if(!trackingIsGood)
	{
		relocalizer.updateCurrentFrame(trackingNewFrame);

		unmappedTrackedFramesMutex.lock();
		unmappedTrackedFramesSignal.notify_one();
		unmappedTrackedFramesMutex.unlock();
		return;
	}

	currentKeyFrameMutex.lock();
	bool my_createNewKeyframe = createNewKeyFrame;	// pre-save here, to make decision afterwards.
	if(trackingReference->keyframe != currentKeyFrame.get() || currentKeyFrame->depthHasBeenUpdatedFlag)
	{
		trackingReference->importFrame(currentKeyFrame.get());
		currentKeyFrame->depthHasBeenUpdatedFlag = false;
		trackingReferenceFrameSharedPT = currentKeyFrame;
	}

	FramePoseStruct* trackingReferencePose = trackingReference->keyframe->pose;
	currentKeyFrameMutex.unlock();

	// DO TRACKING & Show tracking result.
	if(enablePrintDebugInfo && printThreadingInfo)
		printf("TRACKING %d on %d\n", trackingNewFrame->id(), trackingReferencePose->frameID);


	poseConsistencyMutex.lock_shared();
	SE3 frameToReference_initialEstimate = se3FromSim3(
			trackingReferencePose->getCamToWorld().inverse() * keyFrameGraph->allFramePoses.back()->getCamToWorld());
	poseConsistencyMutex.unlock_shared();



	struct timeval tv_start, tv_end;
	gettimeofday(&tv_start, NULL);

	SE3 newRefToFrame_poseUpdate = tracker->trackFrame(
			trackingReference,
			trackingNewFrame.get(),
			frameToReference_initialEstimate);
        /**/
        if(1 == 1)
        {
        getCurentDepthMap(trackingNewFrame.get(), newRefToFrame_poseUpdate,  trackingReference,(float*)sensorDepth.data);
                
         const float* color = trackingNewFrame.get()->image(0);
        curImage =  color;
        if(outputWrapper!= 0){
		//outputWrapper->publishKeyframe(currentKeyFrame.get(),sensorDepth_,curDepth,curImage,newRefToFrame_poseUpdate,correctedDepth,prevCorrectedDepth);
                  outputWrapper->publishKeyframe(currentKeyFrame.get(),sensorDepth_,curDepth,curImage,newRefToFrame_poseUpdate,correctedDepth,prevCorrectedDepth,trackingNewFrame.get());
           
        }
        }
        
	gettimeofday(&tv_end, NULL);
	msTrackFrame = 0.9*msTrackFrame + 0.1*((tv_end.tv_sec-tv_start.tv_sec)*1000.0f + (tv_end.tv_usec-tv_start.tv_usec)/1000.0f);
	nTrackFrame++;

	tracking_lastResidual = tracker->lastResidual;
	tracking_lastUsage = tracker->pointUsage;
	tracking_lastGoodPerBad = tracker->lastGoodCount / (tracker->lastGoodCount + tracker->lastBadCount);
	tracking_lastGoodPerTotal = tracker->lastGoodCount / (trackingNewFrame->width(SE3TRACKING_MIN_LEVEL)*trackingNewFrame->height(SE3TRACKING_MIN_LEVEL));


	if(manualTrackingLossIndicated || tracker->diverged || (keyFrameGraph->keyframesAll.size() > INITIALIZATION_PHASE_COUNT && !tracker->trackingWasGood))
	{
		printf("TRACKING LOST for frame %d (%1.2f%% good Points, which is %1.2f%% of available points, %s)!\n",
				trackingNewFrame->id(),
				100*tracking_lastGoodPerTotal,
				100*tracking_lastGoodPerBad,
				tracker->diverged ? "DIVERGED" : "NOT DIVERGED");

		trackingReference->invalidate();

		trackingIsGood = false;
		nextRelocIdx = -1;

		unmappedTrackedFramesMutex.lock();
		unmappedTrackedFramesSignal.notify_one();
		unmappedTrackedFramesMutex.unlock();

		manualTrackingLossIndicated = false;
		return;
	}



	if(plotTracking)
	{
		Eigen::Matrix<float, 20, 1> data;
		data.setZero();
		data[0] = tracker->lastResidual;

		data[3] = tracker->lastGoodCount / (tracker->lastGoodCount + tracker->lastBadCount);
		data[4] = 4*tracker->lastGoodCount / (width*height);
		data[5] = tracker->pointUsage;

		data[6] = tracker->affineEstimation_a;
		data[7] = tracker->affineEstimation_b;
		outputWrapper->publishDebugInfo(data);
	}

	keyFrameGraph->addFrame(trackingNewFrame.get());

        currentFrame = newRefToFrame_poseUpdate;
	//Sim3 lastTrackedCamToWorld = mostCurrentTrackedFrame->getScaledCamToWorld();//  mostCurrentTrackedFrame->TrackingParent->getScaledCamToWorld() * sim3FromSE3(mostCurrentTrackedFrame->thisToParent_SE3TrackingResult, 1.0);
	if (outputWrapper != 0)
	{
		outputWrapper->publishTrackedFrame(trackingNewFrame.get(),sensorDepth_);
                //cv::Mat valid = cv::Mat(map->depth.rows , map->depth.cols , CV_8UC1, map->currentDepthMap->isValid).clone();
                map->depth.setTo(0,map->valid == 0);// map->depth * valid;
                outputWrapper->publishDepth(map->depth);
	}
         if(0 ==1)
        {
        getCurentDepthMap(trackingNewFrame.get(), newRefToFrame_poseUpdate,  trackingReference,(float*)sensorDepth.data);
                
         const float* color = trackingNewFrame.get()->image(0);
        curImage =  color;
        if(outputWrapper!= 0)
	{	
            outputWrapper->publishKeyframe(currentKeyFrame.get(),sensorDepth_,curDepth,curImage,newRefToFrame_poseUpdate,correctedDepth,prevCorrectedDepth,trackingNewFrame.get());
             //outputWrapper->publishKeyframe(currentKeyFrame.get(),(float*)currentKeyFrame.get()->sensordepth.data,curDepth,curImage,currentFrame,correctedDepth,prevCorrectedDepth,currentKeyFrame.get())
        }
        }

	// Keyframe selection
	latestTrackedFrame = trackingNewFrame;
	if (!my_createNewKeyframe && currentKeyFrame->numMappedOnThisTotal > MIN_NUM_MAPPED)
	{
		Sophus::Vector3d dist = newRefToFrame_poseUpdate.translation() * currentKeyFrame->meanIdepth;
		float minVal = fmin(0.2f + keyFrameGraph->keyframesAll.size() * 0.8f / INITIALIZATION_PHASE_COUNT,1.0f);

		if(keyFrameGraph->keyframesAll.size() < INITIALIZATION_PHASE_COUNT)	minVal *= 0.7;

		lastTrackingClosenessScore = trackableKeyFrameSearch->getRefFrameScore(dist.dot(dist), tracker->pointUsage);

		if (lastTrackingClosenessScore > minVal)
		{
			createNewKeyFrame = true;

			if(enablePrintDebugInfo && printKeyframeSelectionInfo)
				printf("SELECT %d on %d! dist %.3f + usage %.3f = %.3f > 1\n",trackingNewFrame->id(),trackingNewFrame->getTrackingParent()->id(), dist.dot(dist), tracker->pointUsage, trackableKeyFrameSearch->getRefFrameScore(dist.dot(dist), tracker->pointUsage));
		}
		else
		{
			if(enablePrintDebugInfo && printKeyframeSelectionInfo)
				printf("SKIPPD %d on %d! dist %.3f + usage %.3f = %.3f > 1\n",trackingNewFrame->id(),trackingNewFrame->getTrackingParent()->id(), dist.dot(dist), tracker->pointUsage, trackableKeyFrameSearch->getRefFrameScore(dist.dot(dist), tracker->pointUsage));

		}
	}


	unmappedTrackedFramesMutex.lock();
	if(unmappedTrackedFrames.size() < 50 || (unmappedTrackedFrames.size() < 100 && trackingNewFrame->getTrackingParent()->numMappedOnThisTotal < 10))
		unmappedTrackedFrames.push_back(trackingNewFrame);
	unmappedTrackedFramesSignal.notify_one();
	unmappedTrackedFramesMutex.unlock();

	// implement blocking
	if(blockUntilMapped && trackingIsGood)
	{
		boost::unique_lock<boost::mutex> lock(newFrameMappedMutex);
		while(unmappedTrackedFrames.size() > 0)
		{
			//printf("TRACKING IS BLOCKING, waiting for %d frames to finish mapping.\n", (int)unmappedTrackedFrames.size());
			newFrameMappedSignal.wait(lock);
		}
		lock.unlock();
	}
}


float SlamSystem::tryTrackSim3(
		TrackingReference* A, TrackingReference* B,
		int lvlStart, int lvlEnd,
		bool useSSE,
		Sim3 &AtoB, Sim3 &BtoA,
		KFConstraintStruct* e1, KFConstraintStruct* e2 )
{
	BtoA = constraintTracker->trackFrameSim3(
			A,
			B->keyframe,
			BtoA,
			lvlStart,lvlEnd);
	Matrix7x7 BtoAInfo = constraintTracker->lastSim3Hessian;
	float BtoA_meanResidual = constraintTracker->lastResidual;
	float BtoA_meanDResidual = constraintTracker->lastDepthResidual;
	float BtoA_meanPResidual = constraintTracker->lastPhotometricResidual;
	float BtoA_usage = constraintTracker->pointUsage;


	if (constraintTracker->diverged ||
		BtoA.scale() > 1 / Sophus::SophusConstants<sophusType>::epsilon() ||
		BtoA.scale() < Sophus::SophusConstants<sophusType>::epsilon() ||
		BtoAInfo(0,0) == 0 ||
		BtoAInfo(6,6) == 0)
	{
		return 1e20;
	}


	AtoB = constraintTracker->trackFrameSim3(
			B,
			A->keyframe,
			AtoB,
			lvlStart,lvlEnd);
	Matrix7x7 AtoBInfo = constraintTracker->lastSim3Hessian;
	float AtoB_meanResidual = constraintTracker->lastResidual;
	float AtoB_meanDResidual = constraintTracker->lastDepthResidual;
	float AtoB_meanPResidual = constraintTracker->lastPhotometricResidual;
	float AtoB_usage = constraintTracker->pointUsage;


	if (constraintTracker->diverged ||
		AtoB.scale() > 1 / Sophus::SophusConstants<sophusType>::epsilon() ||
		AtoB.scale() < Sophus::SophusConstants<sophusType>::epsilon() ||
		AtoBInfo(0,0) == 0 ||
		AtoBInfo(6,6) == 0)
	{
		return 1e20;
	}

	// Propagate uncertainty (with d(a * b) / d(b) = Adj_a) and calculate Mahalanobis norm
	Matrix7x7 datimesb_db = AtoB.cast<float>().Adj();
	Matrix7x7 diffHesse = (AtoBInfo.inverse() + datimesb_db * BtoAInfo.inverse() * datimesb_db.transpose()).inverse();
	Vector7 diff = (AtoB * BtoA).log().cast<float>();


	float reciprocalConsistency = (diffHesse * diff).dot(diff);


	if(e1 != 0 && e2 != 0)
	{
		e1->firstFrame = A->keyframe;
		e1->secondFrame = B->keyframe;
		e1->secondToFirst = BtoA;
		e1->information = BtoAInfo.cast<double>();
		e1->meanResidual = BtoA_meanResidual;
		e1->meanResidualD = BtoA_meanDResidual;
		e1->meanResidualP = BtoA_meanPResidual;
		e1->usage = BtoA_usage;

		e2->firstFrame = B->keyframe;
		e2->secondFrame = A->keyframe;
		e2->secondToFirst = AtoB;
		e2->information = AtoBInfo.cast<double>();
		e2->meanResidual = AtoB_meanResidual;
		e2->meanResidualD = AtoB_meanDResidual;
		e2->meanResidualP = AtoB_meanPResidual;
		e2->usage = AtoB_usage;

		e1->reciprocalConsistency = e2->reciprocalConsistency = reciprocalConsistency;
	}

	return reciprocalConsistency;
}


void SlamSystem::testConstraint(
		Frame* candidate,
		KFConstraintStruct* &e1_out, KFConstraintStruct* &e2_out,
		Sim3 candidateToFrame_initialEstimate,
		float strictness)
{
	candidateTrackingReference->importFrame(candidate);

	Sim3 FtoC = candidateToFrame_initialEstimate.inverse(), CtoF = candidateToFrame_initialEstimate;
	Matrix7x7 FtoCInfo, CtoFInfo;

	float err_level3 = tryTrackSim3(
			newKFTrackingReference, candidateTrackingReference,	// A = frame; b = candidate
			SIM3TRACKING_MAX_LEVEL-1, 3,
			USESSE,
			FtoC, CtoF);

	if(err_level3 > 3000*strictness)
	{
		if(enablePrintDebugInfo && printConstraintSearchInfo)
			printf("FAILE %d -> %d (lvl %d): errs (%.1f / - / -).",
				newKFTrackingReference->frameID, candidateTrackingReference->frameID,
				3,
				sqrtf(err_level3));

		e1_out = e2_out = 0;

		newKFTrackingReference->keyframe->trackingFailed.insert(std::pair<Frame*,Sim3>(candidate, candidateToFrame_initialEstimate));
		return;
	}

	float err_level2 = tryTrackSim3(
			newKFTrackingReference, candidateTrackingReference,	// A = frame; b = candidate
			2, 2,
			USESSE,
			FtoC, CtoF);

	if(err_level2 > 4000*strictness)
	{
		if(enablePrintDebugInfo && printConstraintSearchInfo)
			printf("FAILE %d -> %d (lvl %d): errs (%.1f / %.1f / -).",
				newKFTrackingReference->frameID, candidateTrackingReference->frameID,
				2,
				sqrtf(err_level3), sqrtf(err_level2));

		e1_out = e2_out = 0;
		newKFTrackingReference->keyframe->trackingFailed.insert(std::pair<Frame*,Sim3>(candidate, candidateToFrame_initialEstimate));
		return;
	}

	e1_out = new KFConstraintStruct();
	e2_out = new KFConstraintStruct();


	float err_level1 = tryTrackSim3(
			newKFTrackingReference, candidateTrackingReference,	// A = frame; b = candidate
			1, 1,
			USESSE,
			FtoC, CtoF, e1_out, e2_out);

	if(err_level1 > 6000*strictness)
	{
		if(enablePrintDebugInfo && printConstraintSearchInfo)
			printf("FAILE %d -> %d (lvl %d): errs (%.1f / %.1f / %.1f).",
					newKFTrackingReference->frameID, candidateTrackingReference->frameID,
					1,
					sqrtf(err_level3), sqrtf(err_level2), sqrtf(err_level1));

		delete e1_out;
		delete e2_out;
		e1_out = e2_out = 0;
		newKFTrackingReference->keyframe->trackingFailed.insert(std::pair<Frame*,Sim3>(candidate, candidateToFrame_initialEstimate));
		return;
	}


	if(enablePrintDebugInfo && printConstraintSearchInfo)
		printf("ADDED %d -> %d: errs (%.1f / %.1f / %.1f).",
			newKFTrackingReference->frameID, candidateTrackingReference->frameID,
			sqrtf(err_level3), sqrtf(err_level2), sqrtf(err_level1));


	const float kernelDelta = 5 * sqrt(6000*loopclosureStrictness);
	e1_out->robustKernel = new g2o::RobustKernelHuber();
	e1_out->robustKernel->setDelta(kernelDelta);
	e2_out->robustKernel = new g2o::RobustKernelHuber();
	e2_out->robustKernel->setDelta(kernelDelta);
}

int SlamSystem::findConstraintsForNewKeyFrames(Frame* newKeyFrame, bool forceParent, bool useFABMAP, float closeCandidatesTH)
{
	if(!newKeyFrame->hasTrackingParent())
	{
		newConstraintMutex.lock();
		keyFrameGraph->addKeyFrame(newKeyFrame);
		newConstraintAdded = true;
		newConstraintCreatedSignal.notify_all();
		newConstraintMutex.unlock();
		return 0;
	}

	if(!forceParent && (newKeyFrame->lastConstraintTrackedCamToWorld * newKeyFrame->getScaledCamToWorld().inverse()).log().norm() < 0.01)
		return 0;


	newKeyFrame->lastConstraintTrackedCamToWorld = newKeyFrame->getScaledCamToWorld();

	// =============== get all potential candidates and their initial relative pose. =================
	std::vector<KFConstraintStruct*, Eigen::aligned_allocator<KFConstraintStruct*> > constraints;
	Frame* fabMapResult = 0;
	std::unordered_set<Frame*, std::hash<Frame*>, std::equal_to<Frame*>,
		Eigen::aligned_allocator< Frame* > > candidates = trackableKeyFrameSearch->findCandidates(newKeyFrame, fabMapResult, useFABMAP, closeCandidatesTH);
	std::map< Frame*, Sim3, std::less<Frame*>, Eigen::aligned_allocator<std::pair<Frame*, Sim3> > > candidateToFrame_initialEstimateMap;


	// erase the ones that are already neighbours.
	for(std::unordered_set<Frame*>::iterator c = candidates.begin(); c != candidates.end();)
	{
		if(newKeyFrame->neighbors.find(*c) != newKeyFrame->neighbors.end())
		{
			if(enablePrintDebugInfo && printConstraintSearchInfo)
				printf("SKIPPING %d on %d cause it already exists as constraint.\n", (*c)->id(), newKeyFrame->id());
			c = candidates.erase(c);
		}
		else
			++c;
	}

	poseConsistencyMutex.lock_shared();
	for (Frame* candidate : candidates)
	{
		Sim3 candidateToFrame_initialEstimate = newKeyFrame->getScaledCamToWorld().inverse() * candidate->getScaledCamToWorld();
		candidateToFrame_initialEstimateMap[candidate] = candidateToFrame_initialEstimate;
	}

	std::unordered_map<Frame*, int> distancesToNewKeyFrame;
	if(newKeyFrame->hasTrackingParent())
		keyFrameGraph->calculateGraphDistancesToFrame(newKeyFrame->getTrackingParent(), &distancesToNewKeyFrame);
	poseConsistencyMutex.unlock_shared();





	// =============== distinguish between close and "far" candidates in Graph =================
	// Do a first check on trackability of close candidates.
	std::unordered_set<Frame*, std::hash<Frame*>, std::equal_to<Frame*>,
		Eigen::aligned_allocator< Frame* > > closeCandidates;
	std::vector<Frame*, Eigen::aligned_allocator<Frame*> > farCandidates;
	Frame* parent = newKeyFrame->hasTrackingParent() ? newKeyFrame->getTrackingParent() : 0;

	int closeFailed = 0;
	int closeInconsistent = 0;

	SO3 disturbance = SO3::exp(Sophus::Vector3d(0.05,0,0));

	for (Frame* candidate : candidates)
	{
		if (candidate->id() == newKeyFrame->id())
			continue;
		if(!candidate->pose->isInGraph)
			continue;
		if(newKeyFrame->hasTrackingParent() && candidate == newKeyFrame->getTrackingParent())
			continue;
		if(candidate->idxInKeyframes < INITIALIZATION_PHASE_COUNT)
			continue;

		SE3 c2f_init = se3FromSim3(candidateToFrame_initialEstimateMap[candidate].inverse()).inverse();
		c2f_init.so3() = c2f_init.so3() * disturbance;
		SE3 c2f = constraintSE3Tracker->trackFrameOnPermaref(candidate, newKeyFrame, c2f_init);
		if(!constraintSE3Tracker->trackingWasGood) {closeFailed++; continue;}


		SE3 f2c_init = se3FromSim3(candidateToFrame_initialEstimateMap[candidate]).inverse();
		f2c_init.so3() = disturbance * f2c_init.so3();
		SE3 f2c = constraintSE3Tracker->trackFrameOnPermaref(newKeyFrame, candidate, f2c_init);
		if(!constraintSE3Tracker->trackingWasGood) {closeFailed++; continue;}

		if((f2c.so3() * c2f.so3()).log().norm() >= 0.09) {closeInconsistent++; continue;}

		closeCandidates.insert(candidate);
	}



	int farFailed = 0;
	int farInconsistent = 0;
	for (Frame* candidate : candidates)
	{
		if (candidate->id() == newKeyFrame->id())
			continue;
		if(!candidate->pose->isInGraph)
			continue;
		if(newKeyFrame->hasTrackingParent() && candidate == newKeyFrame->getTrackingParent())
			continue;
		if(candidate->idxInKeyframes < INITIALIZATION_PHASE_COUNT)
			continue;

		if(candidate == fabMapResult)
		{
			farCandidates.push_back(candidate);
			continue;
		}

		if(distancesToNewKeyFrame.at(candidate) < 4)
			continue;

		farCandidates.push_back(candidate);
	}




	int closeAll = closeCandidates.size();
	int farAll = farCandidates.size();

	// erase the ones that we tried already before (close)
	for(std::unordered_set<Frame*>::iterator c = closeCandidates.begin(); c != closeCandidates.end();)
	{
		if(newKeyFrame->trackingFailed.find(*c) == newKeyFrame->trackingFailed.end())
		{
			++c;
			continue;
		}
		auto range = newKeyFrame->trackingFailed.equal_range(*c);

		bool skip = false;
		Sim3 f2c = candidateToFrame_initialEstimateMap[*c].inverse();
		for (auto it = range.first; it != range.second; ++it)
		{
			if((f2c * it->second).log().norm() < 0.1)
			{
				skip=true;
				break;
			}
		}

		if(skip)
		{
			if(enablePrintDebugInfo && printConstraintSearchInfo)
				printf("SKIPPING %d on %d (NEAR), cause we already have tried it.\n", (*c)->id(), newKeyFrame->id());
			c = closeCandidates.erase(c);
		}
		else
			++c;
	}

	// erase the ones that are already neighbours (far)
	for(unsigned int i=0;i<farCandidates.size();i++)
	{
		if(newKeyFrame->trackingFailed.find(farCandidates[i]) == newKeyFrame->trackingFailed.end())
			continue;

		auto range = newKeyFrame->trackingFailed.equal_range(farCandidates[i]);

		bool skip = false;
		for (auto it = range.first; it != range.second; ++it)
		{
			if((it->second).log().norm() < 0.2)
			{
				skip=true;
				break;
			}
		}

		if(skip)
		{
			if(enablePrintDebugInfo && printConstraintSearchInfo)
				printf("SKIPPING %d on %d (FAR), cause we already have tried it.\n", farCandidates[i]->id(), newKeyFrame->id());
			farCandidates[i] = farCandidates.back();
			farCandidates.pop_back();
			i--;
		}
	}



	if (enablePrintDebugInfo && printConstraintSearchInfo)
		printf("Final Loop-Closure Candidates: %d / %d close (%d failed, %d inconsistent) + %d / %d far (%d failed, %d inconsistent) = %d\n",
				(int)closeCandidates.size(),closeAll, closeFailed, closeInconsistent,
				(int)farCandidates.size(), farAll, farFailed, farInconsistent,
				(int)closeCandidates.size() + (int)farCandidates.size());



	// =============== limit number of close candidates ===============
	// while too many, remove the one with the highest connectivity.
	while((int)closeCandidates.size() > maxLoopClosureCandidates)
	{
		Frame* worst = 0;
		int worstNeighbours = 0;
		for(Frame* f : closeCandidates)
		{
			int neightboursInCandidates = 0;
			for(Frame* n : f->neighbors)
				if(closeCandidates.find(n) != closeCandidates.end())
					neightboursInCandidates++;

			if(neightboursInCandidates > worstNeighbours || worst == 0)
			{
				worst = f;
				worstNeighbours = neightboursInCandidates;
			}
		}

		closeCandidates.erase(worst);
	}


	// =============== limit number of far candidates ===============
	// delete randomly
	int maxNumFarCandidates = (maxLoopClosureCandidates +1) / 2;
	if(maxNumFarCandidates < 5) maxNumFarCandidates = 5;
	while((int)farCandidates.size() > maxNumFarCandidates)
	{
		int toDelete = rand() % farCandidates.size();
		if(farCandidates[toDelete] != fabMapResult)
		{
			farCandidates[toDelete] = farCandidates.back();
			farCandidates.pop_back();
		}
	}







	// =============== TRACK! ===============

	// make tracking reference for newKeyFrame.
	newKFTrackingReference->importFrame(newKeyFrame);


	for (Frame* candidate : closeCandidates)
	{
		KFConstraintStruct* e1=0;
		KFConstraintStruct* e2=0;

		testConstraint(
				candidate, e1, e2,
				candidateToFrame_initialEstimateMap[candidate],
				loopclosureStrictness);

		if(enablePrintDebugInfo && printConstraintSearchInfo)
			printf(" CLOSE (%d)\n", distancesToNewKeyFrame.at(candidate));

		if(e1 != 0)
		{
			constraints.push_back(e1);
			constraints.push_back(e2);

			// delete from far candidates if it's in there.
			for(unsigned int k=0;k<farCandidates.size();k++)
			{
				if(farCandidates[k] == candidate)
				{
					if(enablePrintDebugInfo && printConstraintSearchInfo)
						printf(" DELETED %d from far, as close was successful!\n", candidate->id());

					farCandidates[k] = farCandidates.back();
					farCandidates.pop_back();
				}
			}
		}
	}


	for (Frame* candidate : farCandidates)
	{
		KFConstraintStruct* e1=0;
		KFConstraintStruct* e2=0;

		testConstraint(
				candidate, e1, e2,
				Sim3(),
				loopclosureStrictness);

		if(enablePrintDebugInfo && printConstraintSearchInfo)
			printf(" FAR (%d)\n", distancesToNewKeyFrame.at(candidate));

		if(e1 != 0)
		{
			constraints.push_back(e1);
			constraints.push_back(e2);
		}
	}



	if(parent != 0 && forceParent)
	{
		KFConstraintStruct* e1=0;
		KFConstraintStruct* e2=0;
		testConstraint(
				parent, e1, e2,
				candidateToFrame_initialEstimateMap[parent],
				100);
		if(enablePrintDebugInfo && printConstraintSearchInfo)
			printf(" PARENT (0)\n");

		if(e1 != 0)
		{
			constraints.push_back(e1);
			constraints.push_back(e2);
		}
		else
		{
			float downweightFac = 5;
			const float kernelDelta = 5 * sqrt(6000*loopclosureStrictness) / downweightFac;
			printf("warning: reciprocal tracking on new frame failed badly, added odometry edge (Hacky).\n");

			poseConsistencyMutex.lock_shared();
			constraints.push_back(new KFConstraintStruct());
			constraints.back()->firstFrame = newKeyFrame;
			constraints.back()->secondFrame = newKeyFrame->getTrackingParent();
			constraints.back()->secondToFirst = constraints.back()->firstFrame->getScaledCamToWorld().inverse() * constraints.back()->secondFrame->getScaledCamToWorld();
			constraints.back()->information  <<
					0.8098,-0.1507,-0.0557, 0.1211, 0.7657, 0.0120, 0,
					-0.1507, 2.1724,-0.1103,-1.9279,-0.1182, 0.1943, 0,
					-0.0557,-0.1103, 0.2643,-0.0021,-0.0657,-0.0028, 0.0304,
					 0.1211,-1.9279,-0.0021, 2.3110, 0.1039,-0.0934, 0.0005,
					 0.7657,-0.1182,-0.0657, 0.1039, 1.0545, 0.0743,-0.0028,
					 0.0120, 0.1943,-0.0028,-0.0934, 0.0743, 0.4511, 0,
					0,0, 0.0304, 0.0005,-0.0028, 0, 0.0228;
			constraints.back()->information *= (1e9/(downweightFac*downweightFac));

			constraints.back()->robustKernel = new g2o::RobustKernelHuber();
			constraints.back()->robustKernel->setDelta(kernelDelta);

			constraints.back()->meanResidual = 10;
			constraints.back()->meanResidualD = 10;
			constraints.back()->meanResidualP = 10;
			constraints.back()->usage = 0;

			poseConsistencyMutex.unlock_shared();
		}
	}


	newConstraintMutex.lock();

	keyFrameGraph->addKeyFrame(newKeyFrame);
	for(unsigned int i=0;i<constraints.size();i++)
		keyFrameGraph->insertConstraint(constraints[i]);


	newConstraintAdded = true;
	newConstraintCreatedSignal.notify_all();
	newConstraintMutex.unlock();

	newKFTrackingReference->invalidate();
	candidateTrackingReference->invalidate();



	return constraints.size();
}




bool SlamSystem::optimizationIteration(int itsPerTry, float minChange)
{
	struct timeval tv_start, tv_end;
	gettimeofday(&tv_start, NULL);



	g2oGraphAccessMutex.lock();

	// lock new elements buffer & take them over.
	newConstraintMutex.lock();
	keyFrameGraph->addElementsFromBuffer();
	newConstraintMutex.unlock();


	// Do the optimization. This can take quite some time!
	int its = keyFrameGraph->optimize(itsPerTry);
	

	// save the optimization result.
	poseConsistencyMutex.lock_shared();
	keyFrameGraph->keyframesAllMutex.lock_shared();
	float maxChange = 0;
	float sumChange = 0;
	float sum = 0;
	for(size_t i=0;i<keyFrameGraph->keyframesAll.size(); i++)
	{
		// set edge error sum to zero
		keyFrameGraph->keyframesAll[i]->edgeErrorSum = 0;
		keyFrameGraph->keyframesAll[i]->edgesNum = 0;

		if(!keyFrameGraph->keyframesAll[i]->pose->isInGraph) continue;



		// get change from last optimization
		Sim3 a = keyFrameGraph->keyframesAll[i]->pose->graphVertex->estimate();
		Sim3 b = keyFrameGraph->keyframesAll[i]->getScaledCamToWorld();
		Sophus::Vector7f diff = (a*b.inverse()).log().cast<float>();


		for(int j=0;j<7;j++)
		{
			float d = fabsf((float)(diff[j]));
			if(d > maxChange) maxChange = d;
			sumChange += d;
		}
		sum +=7;

		// set change
		keyFrameGraph->keyframesAll[i]->pose->setPoseGraphOptResult(
				keyFrameGraph->keyframesAll[i]->pose->graphVertex->estimate());

		// add error
		for(auto edge : keyFrameGraph->keyframesAll[i]->pose->graphVertex->edges())
		{
			keyFrameGraph->keyframesAll[i]->edgeErrorSum += ((EdgeSim3*)(edge))->chi2();
			keyFrameGraph->keyframesAll[i]->edgesNum++;
		}
	}

	haveUnmergedOptimizationOffset = true;
	keyFrameGraph->keyframesAllMutex.unlock_shared();
	poseConsistencyMutex.unlock_shared();

	g2oGraphAccessMutex.unlock();

	if(enablePrintDebugInfo && printOptimizationInfo)
		printf("did %d optimization iterations. Max Pose Parameter Change: %f; avgChange: %f. %s\n", its, maxChange, sumChange / sum,
				maxChange > minChange && its == itsPerTry ? "continue optimizing":"Waiting for addition to graph.");


	gettimeofday(&tv_end, NULL);
	msOptimizationIteration = 0.9*msOptimizationIteration + 0.1*((tv_end.tv_sec-tv_start.tv_sec)*1000.0f + (tv_end.tv_usec-tv_start.tv_usec)/1000.0f);
	nOptimizationIteration++;


	return maxChange > minChange && its == itsPerTry;
}

void SlamSystem::optimizeGraph()
{
	boost::unique_lock<boost::mutex> g2oLock(g2oGraphAccessMutex);
	keyFrameGraph->optimize(1000);
	g2oLock.unlock();
	mergeOptimizationOffset();
}


SE3 SlamSystem::getCurrentPoseEstimate()
{
	SE3 camToWorld = SE3();
	keyFrameGraph->allFramePosesMutex.lock_shared();
	if(keyFrameGraph->allFramePoses.size() > 0)
		camToWorld = se3FromSim3(keyFrameGraph->allFramePoses.back()->getCamToWorld());
	keyFrameGraph->allFramePosesMutex.unlock_shared();
	return camToWorld;
}

std::vector<FramePoseStruct*, Eigen::aligned_allocator<FramePoseStruct*> > SlamSystem::getAllPoses()
{
	return keyFrameGraph->allFramePoses;
}
//SlamSystem::

void SlamSystem::projectToNewFrame(float *pyrIdepthSource,Eigen::Matrix3f rotMat,Eigen::Vector3f transVec,int w ,int h, float fx_l, float fy_l , float cx_l, float cy_l, float fxInv, float cxInv, float fyInv , float cyInv)
{
    
    float * transformedDepth = new float[sizeof(float) * w *h];
     //std::cout <<"1 "<<std::endl;
     std::memset(transformedDepth, 0,  w*h *sizeof(float));
     //std::cout <<"2 "<<std::endl;
    for(int x=1; x<w-1; x++){
		for(int y=1; y<h-1; y++)
		{
			int idx = x + y*w;

			if(std::isnan(pyrIdepthSource[idx]) || pyrIdepthSource[idx] <= 0 || pyrIdepthSource[idx] == 0) continue;

			Eigen::Vector3f  posDataPT = ( pyrIdepthSource[idx]) * Eigen::Vector3f(fxInv *x+cxInv,fyInv *y+cyInv,1);
               
                       Eigen::Vector3f Wxp = rotMat * (posDataPT) + transVec;
                
                        float u_new = (Wxp[0]/Wxp[2])*fx_l + cx_l;
                        float v_new = (Wxp[1]/Wxp[2])*fy_l + cy_l;
                        if(!(u_new > 0 && v_new > 0 && u_new < w && v_new < h))
                        {
			
                                continue;
                        }
                
                        int id = int(u_new)  + (int(v_new)* w);
                        
                        transformedDepth[id]= Wxp[2];
                    }
    }
    // std::cout <<"3 "<<std::endl;
    
    //std::cout <<"4 "<<std::endl;
    //pyrIdepthSource = transformedDepth;
     std::memcpy(pyrIdepthSource,transformedDepth,sizeof(float) * w * h);
    delete [] transformedDepth;
    //std::cout <<"5 "<<std::endl;
}
 

float computeNewMean(float mean1 , float var1 , float mean2, float var2)
{

    float nMean = ((mean1 * var2) + (mean2 * var1))/(var1 + var2);
    return nMean;
}
float computeNewVar(float var1 , float var2){
    float nVar = (var1 * var2) / (var1 + var2);
    return nVar;

}

void SlamSystem::getCurentDepthMap(Frame* frame ,  const Sophus::SE3& referenceToFrame_0, TrackingReference* reference, float* sensorDepth)
{
        //correctedDepth;
        //prevCorrectedFrame;
        /*
        if(frame->id == 1){
        //prevCorrectedDepth.reset(new Frame(1, width, height, K, frame->timeStamp, image));
              //prevCorrectedDepth  = new TrackingReference();
           }
        */
        int level = 0;//QUICK_KF_CHECK_LVL;
        
        int w = frame->width(level);
	int h = frame->height(level);
        
        
        //std::cout << " 1 "<<std::endl; 
        std::memset(curDepth, 0,  w*h *sizeof(float));
      
        std::memset(correctedDepth, 0,  w*h *sizeof(float));
        
      
	Eigen::Matrix3f KLvl = frame->K(level);
	float fx_l = KLvl(0,0);
	float fy_l = KLvl(1,1);
	float cx_l = KLvl(0,2);
	float cy_l = KLvl(1,2);
        
        Eigen::Matrix3f KInvLvl = frame->KInv(level);
        
        float fxInv =  KInvLvl(0,0);
        float cxInv =  KInvLvl(1,1);
        float fyInv =  KInvLvl(0,2);
        float cyInv =  KInvLvl(1,2); 
        
       
        reference->makePointCloud(level);
        const Eigen::Vector3f* refPoint = reference->posData[level];
        int refNum = reference->numData[level];
	const Eigen::Vector3f* refPoint_max = refPoint + refNum;
       
        Sophus::SE3f referenceToFrame = referenceToFrame_0.inverse().cast<float>();
	
         
	Eigen::Matrix3f rotMat = referenceToFrame.rotationMatrix();
	Eigen::Vector3f transVec = referenceToFrame.translation();
        
        const float* color = frame->image(level);
        curImage =  color;
        std::vector<cv::Point3f> objectPoints;
        std::vector<cv::Point2f> imagePoints;
        
        const float* var = reference->keyframe->idepthVar(level);
        /*
        
	//std::cout <<"widt height "<<w<< " "<<h <<std::endl;
        int u = 0;
        int v = 0;
        int scaleCount = 0;
        float sensorSum = 0;
        float lsdSum = 0;
        //float scale = 0;
        
        
	for(;refPoint<refPoint_max; refPoint++)
	{
                
		Eigen::Vector3f Wxp = rotMat * (*refPoint) + transVec;
                
		float u_new = (Wxp[0]/Wxp[2])*fx_l + cx_l;
		float v_new = (Wxp[1]/Wxp[2])*fy_l + cy_l;
                if(!(u_new > 0 && v_new > 0 && u_new < w && v_new < h))
		{
			
			continue;
		}
                
                int id = int(u_new)  + (int(v_new)* w);
                
                //int id_orig = u + ( v * w);
                
                //std::cout <<u_new<<" "<<v_new<<" "<<id<<" "<<Wxp[2]<<std::endl;
                
                //curDepth[id]= Wxp[2];
                
                //curImage[id_orig] = color[id_orig];
                 //std::cout << " 2.5 "<<std::endl;
                 // std::cout << id <<" "<<!std::isnan(sensorDepth[id]) <<std::endl;
                //compute scale
                if((!std::isnan(sensorDepth[id])) && sensorDepth[id] > 0 && Wxp[2]> 0){
                  //std::cout << id << std::endl;
                    scaleCount++;
                sensorSum +=sensorDepth[id];
                lsdSum += Wxp[2];
                    
                }
                    
                
                
                //objectPoints.push_back(cv::Point3f(Wxp[0],Wxp[1],Wxp[2]));
        }
        
        float scale = sensorSum / lsdSum;
         */
        float scale = 1 / reference->keyframe->getScaledCamToWorld().scale();;
       // std::cout <<"scale is " << scale << std::endl;
        float sensorDepthMultiplyer =  0.001425;
        refPoint = reference->posData[level];
        std::memcpy(correctedDepth, sensorDepth,  w*h *sizeof(float));
        //& !std::isinf(scale)
        //if(scale != 0   &  !std::isnan(scale) )
        { 
            //std::cout <<"scale is " << scale << std::endl;
        for(;refPoint<refPoint_max; refPoint++)
	{
            Eigen::Vector3f Wxp ;
            //if (fuseAllframes)
		 Wxp = (rotMat * (1 *  (*refPoint))) + transVec;
            //else
             //   Wxp = (scale *  (*refPoint));
            
		float u_new = (Wxp[0]/Wxp[2])*fx_l + cx_l;
		float v_new = (Wxp[1]/Wxp[2])*fy_l + cy_l;
                if(!(u_new > 0 && v_new > 0 && u_new < w && v_new < h))
		{
			
			continue;
		}
                
                int id = int(u_new)  + (int(v_new)* w);
                
                curDepth[id]= Wxp[2];
                float lsdDepth = curDepth[id];
                float sensDepth = sensorDepth[id];
                float senV =  sensorDepthMultiplyer * sensDepth * sensDepth;;
                float lsdV = var[id];
                
                //& !std::isinf(sensDepth) 
                 if((!std::isnan(sensDepth)  & sensDepth  >0)){
                              correctedDepth[id] = sensDepth;
                               float er = (sensDepth - lsdDepth) * (sensDepth - lsdDepth);
                             if((!std::isnan(lsdDepth) & !std::isinf(lsdDepth) & lsdDepth  >0 & er < 1)){
                                 
                                  float newVar = computeNewVar( senV ,  lsdV);
                                  float newDepth =  computeNewMean( sensDepth ,  senV , lsdDepth,  lsdV) ;
                                 
                                  correctedDepth[id] = newDepth;
                                  //fusedVar(w,h) = newVar;
                                  }
                          
                 } //&  !std::isinf(lsdDepth)
                                else if ((!std::isnan(lsdDepth))  & lsdDepth  >0 ){
                             correctedDepth[id] = lsdDepth;
                              //fusedVar(w,h) = lsdvar;
                              }
                
                //std::cout <<"here " << correctedDepth[id]  << std::endl;
         }
        }
        
      
        
}



/*
void SlamSystem::getCurentDepthMap(Frame* frame ,  const Sophus::SE3& referenceToFrame_0, TrackingReference* reference, float* sensorDepth)
{
        //correctedDepth;
        //prevCorrectedFrame;
        /*
        if(frame->id == 1){
        //prevCorrectedDepth.reset(new Frame(1, width, height, K, frame->timeStamp, image));
              //prevCorrectedDepth  = new TrackingReference();
           }
        * /
        int level = 0;//QUICK_KF_CHECK_LVL;
        
        int w = frame->width(level);
	int h = frame->height(level);
        
        
        //std::cout << " 1 "<<std::endl; 
        std::memset(curDepth, 0,  w*h *sizeof(float));
      
        std::memset(correctedDepth, 0,  w*h *sizeof(float));
        
        /*
        if(frame->id() == 1){
            for(int i = 0 ; i < w *h ;i ++){
                prevCorrectedDepth[i] = 1/trackingReference->keyframe[i];
            }
            return;
        }
        * /
        //std::cout << " 2 "<<std::endl;
        //std::memset(curImage, 0,  w*h *sizeof(float));
        
	Eigen::Matrix3f KLvl = frame->K(level);
	float fx_l = KLvl(0,0);
	float fy_l = KLvl(1,1);
	float cx_l = KLvl(0,2);
	float cy_l = KLvl(1,2);
        
        Eigen::Matrix3f KInvLvl = frame->KInv(level);
        
        float fxInv =  KInvLvl(0,0);
        float cxInv =  KInvLvl(1,1);
        float fyInv =  KInvLvl(0,2);
        float cyInv =  KInvLvl(1,2); 
        
       
        reference->makePointCloud(level);
        const Eigen::Vector3f* refPoint = reference->posData[level];
        int refNum = reference->numData[level];
	const Eigen::Vector3f* refPoint_max = refPoint + refNum;
        Eigen::Matrix3f rotMat ;
	Eigen::Vector3f transVec;
        if (fuseAllframes){
        Sophus::SE3f referenceToFrame = referenceToFrame_0.inverse().cast<float>();
	
         
	Eigen::Matrix3f rotMat = referenceToFrame.rotationMatrix();
	Eigen::Vector3f transVec = referenceToFrame.translation();
        }
        const float* color = frame->image(level);
        curImage =  color;
        std::vector<cv::Point3f> objectPoints;
        std::vector<cv::Point2f> imagePoints;
        
        const float* var = reference->keyframe->idepthVar(level);
        /*
        
	//std::cout <<"widt height "<<w<< " "<<h <<std::endl;
        int u = 0;
        int v = 0;
        int scaleCount = 0;
        float sensorSum = 0;
        float lsdSum = 0;
        //float scale = 0;
        
        
	for(;refPoint<refPoint_max; refPoint++)
	{
                
		Eigen::Vector3f Wxp = rotMat * (*refPoint) + transVec;
                
		float u_new = (Wxp[0]/Wxp[2])*fx_l + cx_l;
		float v_new = (Wxp[1]/Wxp[2])*fy_l + cy_l;
                if(!(u_new > 0 && v_new > 0 && u_new < w && v_new < h))
		{
			
			continue;
		}
                
                int id = int(u_new)  + (int(v_new)* w);
                
                //int id_orig = u + ( v * w);
                
                //std::cout <<u_new<<" "<<v_new<<" "<<id<<" "<<Wxp[2]<<std::endl;
                
                //curDepth[id]= Wxp[2];
                
                //curImage[id_orig] = color[id_orig];
                 //std::cout << " 2.5 "<<std::endl;
                 // std::cout << id <<" "<<!std::isnan(sensorDepth[id]) <<std::endl;
                //compute scale
                if((!std::isnan(sensorDepth[id])) && sensorDepth[id] > 0 && Wxp[2]> 0){
                  //std::cout << id << std::endl;
                    scaleCount++;
                sensorSum +=sensorDepth[id];
                lsdSum += Wxp[2];
                    
                }
                    
                
                
                //objectPoints.push_back(cv::Point3f(Wxp[0],Wxp[1],Wxp[2]));
        }
        
        float scale = sensorSum / lsdSum;
         * /
        float scale = globalScale;
       // std::cout <<"scale is " << scale << std::endl;
        float sensorDepthMultiplyer =  0.001425;
        refPoint = reference->posData[level];
        std::memcpy(correctedDepth, sensorDepth,  w*h *sizeof(float));
        //& !std::isinf(scale)
        if(scale != 0   &  !std::isnan(scale) )
        { 
        for(;refPoint<refPoint_max; refPoint++)
	{
            Eigen::Vector3f Wxp ;
            if (fuseAllframes)
		 Wxp = (rotMat * (scale *  (*refPoint))) + transVec;
            else
                Wxp = (scale *  (*refPoint));
            
		float u_new = (Wxp[0]/Wxp[2])*fx_l + cx_l;
		float v_new = (Wxp[1]/Wxp[2])*fy_l + cy_l;
                if(!(u_new > 0 && v_new > 0 && u_new < w && v_new < h))
		{
			
			continue;
		}
                
                int id = int(u_new)  + (int(v_new)* w);
                
                curDepth[id]= Wxp[2];
                float lsdDepth = curDepth[id];
                float sensDepth = sensorDepth[id];
                float senV =  sensorDepthMultiplyer * sensDepth * sensDepth;;
                float lsdV = var[id];
                
                //& !std::isinf(sensDepth) 
                 if((!std::isnan(sensDepth)  & sensDepth  >0)){
                              correctedDepth[id] = sensDepth;
                               float er = (sensDepth - lsdDepth) * (sensDepth - lsdDepth);
                             if((!std::isnan(lsdDepth) & !std::isinf(lsdDepth) & lsdDepth  >0 & er < 1)){
                                 
                                  float newVar = computeNewVar( senV ,  lsdV);
                                  float newDepth =  computeNewMean( sensDepth ,  senV , lsdDepth,  lsdV) ;
                                 
                                  correctedDepth[id] = newDepth;
                                  //fusedVar(w,h) = newVar;
                                  }
                          
                 } //&  !std::isinf(lsdDepth)
                                else if ((!std::isnan(lsdDepth))  & lsdDepth  >0 ){
                             correctedDepth[id] = lsdDepth;
                              //fusedVar(w,h) = lsdvar;
                              }
                
                //std::cout <<"here " << correctedDepth[id]  << std::endl;
         }
        }
        
        /*
        
        float maxVar = 100;
         for(int id = 0 ; id < w *h ;id++){
            //std::cout <<  curDepth[id] << std::endl;
         //std::cout << (!std::isnan(sensorDepth[id]))<<" "<< (sensorDepth[id] > 0) <<" "<<(curDepth[id]> 0)<<" "<<((!std::isnan(sensorDepth[id])) && sensorDepth[id] > 0 && curDepth[id]> 0)<<std::endl;    
        if((!std::isnan(sensorDepth[id])) && sensorDepth[id] > 0 && curDepth[id]> 0){
                  //std::cout << id << std::endl;
                scaleCount += 1;
                sensorSum +=sensorDepth[id];
                lsdSum += curDepth[id];
               // std::cout << "scale cout "<<scaleCount << std::endl;
        }
      }
        
          //std::cout << " 2.6 "<<std::endl;
          if(scaleCount!=0){
           scale =    (sensorSum/lsdSum) ;
          }
        //std::cout <<"scale is " <<scaleCount<<" "<< lsdSum<< " "<<sensorSum<<" "<<scale << std::endl;
        
        
        
        const float* curVar = reference->keyframe->idepthVar(level);
        //std::cout << " 3 "<<std::endl;
         if(!prevDepthCorrectionAvilable){
             //std::cout << " 4 "<<std::endl;
                   std::memset(prevCorrectedDepth, 0,  w*h *sizeof(float));
                 prevDepthCorrectionAvilable = !prevDepthCorrectionAvilable;
                 for(int i = 0 ; i < w *h ;i ++){
                    // if(!std::isnan(sensorDepth[i])){
                       prevCorrectedDepth[i] = sensorDepth[i];
                       prevCorrectVar[i] = 0.4;
                       
                        correctedDepth[i] = sensorDepth[i];
                      correctedDepthVar[i] = 0.1;
                     /*}else{
                     
                       prevCorrectedDepth[i] = 0;
                       prevCorrectVar[i] = maxVar;
                       
                        correctedDepth[i] = 0;
                      correctedDepthVar[i] = maxVar;
                     }* /
                       
                      
                 }
         }else{
             
            //projectToNewFrame( prevCorrectedDepth, rotMat, transVec ,w,h, fx_l, fy_l ,  cx_l,  cy_l, fxInv , cxInv , fyInv, cyInv);
             
             //std::cout << " 6 "<<prevCorrectedDepth[7]<<std::endl;
            //std::cout << " 1 "<<std::endl;
      
                for(int i = 0 ; i < w *h ;i ++){
                     //curDepth[i];
                     //prevCorrectedDepth[i]; 
                     //sensorDepth[i]; 
                     //curVar[i]
                     //prevCorrectVar[i]
                     //sensorVar;
                    
                    //if there is sensor mesurement and if error with previous measumrent 
                    //is less
                    //save for replacing later
                     float corDe = correctedDepth[i];
                     float corVar = correctedDepthVar[i];
                     float sensorVar = maxVar;//0.1
                     int imageSize = w * h;
                     
                       correctedDepth[i] = 0 ;
                       correctedDepthVar[i] =  maxVar;
                       //std::cout << " 2 "<<std::endl;
      
                    if(!std::isnan(sensorDepth[i]) & !std::isinf(sensorDepth[i])& sensorDepth[i] > 0){
                        
                        //std::cout << " 3 "<<std::endl;
      
                        float depthError = std::abs(sensorDepth[i] -prevCorrectedDepth[i] );
                        //also chekc pose change for error
                        //using windowing system instead of single value 
                        
                        //compute spatial variance for sensor depth
                        if(((i - w - 1) >= 0 & (i + w + 1 < imageSize)  )){
                            //std::cout << i << std::endl;
                            float c  = (!std::isnan(sensorDepth[i]))?sensorDepth[i]:0;
                            float n  = (!std::isnan(sensorDepth[i - w]))?sensorDepth[i - w]:0 ;
                            float s  = (!std::isnan(sensorDepth[i + w]))?sensorDepth[i + w]:0 ;
                            float w  = (!std::isnan(sensorDepth[i - 1]))?sensorDepth[i - 1]:0 ;
                            float e  = (!std::isnan(sensorDepth[i + 1]))?sensorDepth[i + 1]:0 ;
                            float ne = (!std::isnan(sensorDepth[int(i - w + 1)]))?sensorDepth[int(i - w + 1)]:0 ;
                            float nw = (!std::isnan(sensorDepth[int(i - w - 1)]))?sensorDepth[int(i - w - 1)]:0 ;
                            float se = (!std::isnan(sensorDepth[int(i + w + 1)]))?sensorDepth[int(i + w + 1)]:0 ;
                            float sw = (!std::isnan(sensorDepth[int(i + w - 1)]))?sensorDepth[int(i + w - 1)]:0 ;
                            
                            float mean = (c + n + s + w + e + ne + nw + se + sw) /9; 
                            float var = maxVar;
                            if(mean != 0){
                               var = (pow((c - mean),2) + pow((n- mean),2) + pow((s- mean),2) + pow((w- mean),2) + pow((e- mean),2) + pow((ne- mean),2) + pow((nw - mean),2)+ pow((se - mean),2)+ pow((sw- mean),2)) /9;
                            }
                            sensorVar = var;
                        
                        }
                        //std::cout << " 4 "<<std::endl;
      
                        if(sensorVar <= 2 & depthError < 2.0){
                            correctedDepth[i] = sensorDepth[i];
                            correctedDepthVar[i] = sensorVar;
                        }
                        else if ( sensorVar != maxVar & depthError < 2.0 ){
                        
                            correctedDepth[i] = computeNewMean(sensorDepth[i], sensorVar , prevCorrectedDepth[i], prevCorrectVar[i]);
                            correctedDepthVar[i] = computeNewVar(sensorVar, prevCorrectVar[i]);
                        }
                        //std::cout << " 5 "<<std::endl;
      
                        
                    }
                       
                     if(sensorVar == maxVar)   {
                    //std::cout << " 6 "<<std::endl;
      
                    if(!std::isnan(curDepth[i]) & !std::isinf(curDepth[i])& curDepth[i] > 0){
                        float scaledDepth = curDepth[i] * scale;
                        curDepth[i] = scaledDepth;
                        float depthError = std::abs(scaledDepth -prevCorrectedDepth[i] );
                        //also chekc pose change for error
                        //using windowing system instead of single value 
                         if(depthError < 2.0){
                            correctedDepth[i] = scaledDepth;
                            correctedDepthVar[i] = curVar[i];
                        }else{
                        
                            correctedDepth[i] = computeNewMean(scaledDepth, curVar[i] , prevCorrectedDepth[i], prevCorrectVar[i]);
                            correctedDepthVar[i] = computeNewVar(curVar[i], prevCorrectVar[i]);
                        }
                        //std::cout << " 7 "<<std::endl;
      
                    }
                     
                      
                     }
                     /*
                     else{
                         correctedDepth[i] = 0;
                         correctedDepthVar[i] = maxVar;
                     
                     
                     }* /
                       prevCorrectedDepth[i] = corDe;
                       //prevCorrectVar[i] = corVar;
                }
             //std::cout << " 8 "<<std::endl;
        }
        */
       
         /*
        cv::Mat intrisicMat(3, 3, cv::DataType<double>::type); // Intrisic matrix
        intrisicMat.at<double>(0, 0) = fx_l;
        intrisicMat.at<double>(1, 0) = 0;
        intrisicMat.at<double>(2, 0) = 0;

        intrisicMat.at<double>(0, 1) = 0;
        intrisicMat.at<double>(1, 1) = fy_l;
        intrisicMat.at<double>(2, 1) = 0;

        intrisicMat.at<double>(0, 2) = cx_l;
        intrisicMat.at<double>(1, 2) = cy_l;
        intrisicMat.at<double>(2, 2) = 1;
        
        cv::Mat rMat(3, 3, cv::DataType<double>::type); // Rotation vector
        
        rMat.at<double>(0,0) = rotMat(0,0);
        rMat.at<double>(0,1) = rotMat(0,1);
        rMat.at<double>(0,2) = rotMat(0,2);
        
        rMat.at<double>(1,0) = rotMat(1,0);
        rMat.at<double>(1,1) = rotMat(1,1);
        rMat.at<double>(1,2) = rotMat(1,2);
        
        rMat.at<double>(2,0) = rotMat(2,0);
        rMat.at<double>(2,1) = rotMat(2,1);
        rMat.at<double>(2,2) = rotMat(2,2);
        cv::Mat rVec(3, 1, cv::DataType<double>::type); // Rotation vector
        Rodrigues(rMat, rVec);

      
      

        cv::Mat tVec(3, 1, cv::DataType<double>::type); // Translation vector
        tVec.at<double>(0) = transVec[0];
        tVec.at<double>(1) = transVec[1];
        tVec.at<double>(2) = transVec[2];
        
        cv::Mat distCoeffs(5, 1, cv::DataType<double>::type); 
        
        cv::Mat imagePoints_mat;
        cv::projectPoints(objectPoints, rVec, tVec, intrisicMat, distCoeffs, imagePoints);
        
        for(int i = 0 ; i < imagePoints.size();i++){
           // std::cout << imagePoints[i].x <<" " << imagePoints[i].y << std::endl;
            float u_new = imagePoints[i].x;
            float v_new = imagePoints[i].y;
            if(!(u_new > 0 && v_new > 0 && u_new < w && v_new < h))
		{
			
			continue;
		}
                
                int id = int(u_new)  + (int(v_new)* w);
                
                //std::cout <<u_new<<" "<<v_new<<" "<<id<<" "<<Wxp[2]<<std::endl;
                
                curDepth[id]= objectPoints[i].z;
        }
          */ 
        /*
        //Ray tracing
        Eigen::Vector3f origin << 0,0,0;
        Eigen::Vector3f direction ;
        float aspectRatio = float(w)/float(h);
        for(int r = 0 ; r < h; r++){
            for(int c = 0 ; c < w ; c++){
                float xx = (2 * (c + 0.5 ) /w - 1)  * aspectRatio;
                float yy = (2 * (r + 0.5) / h - 1);
                direction << (xx,yy,-4);
            }
        }
         * / 
        
}
*/