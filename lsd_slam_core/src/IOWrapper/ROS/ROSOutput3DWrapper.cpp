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

#include "ROSOutput3DWrapper.h"
#include "util/SophusUtil.h"
#include <ros/ros.h>
#include "std_msgs/Float32.h"
#include "util/settings.h"


#include "std_msgs/Float32MultiArray.h"
#include "lsd_slam_viewer/keyframeGraphMsg.h"
#include "lsd_slam_viewer/keyframeMsg.h"

#include "DataStructures/Frame.h"
#include "GlobalMapping/KeyFrameGraph.h"
#include "sophus/sim3.hpp"
#include "geometry_msgs/PoseStamped.h"
#include "GlobalMapping/g2oTypeSim3Sophus.h"


#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include <math.h>

namespace lsd_slam
{


ROSOutput3DWrapper::ROSOutput3DWrapper(int width, int height)
{
	this->width = width;
	this->height = height;

	liveframe_channel = nh_.resolveName("lsd_slam/liveframes");
	liveframe_publisher = nh_.advertise<lsd_slam_viewer::keyframeMsg>(liveframe_channel,1);

	keyframe_channel = nh_.resolveName("lsd_slam/keyframes");
	keyframe_publisher = nh_.advertise<lsd_slam_viewer::keyframeMsg>(keyframe_channel,1);

	graph_channel = nh_.resolveName("lsd_slam/graph");
	graph_publisher = nh_.advertise<lsd_slam_viewer::keyframeGraphMsg>(graph_channel,1);

	debugInfo_channel = nh_.resolveName("lsd_slam/debug");
	debugInfo_publisher = nh_.advertise<std_msgs::Float32MultiArray>(debugInfo_channel,1);

	pose_channel = nh_.resolveName("lsd_slam/pose");
	pose_publisher = nh_.advertise<geometry_msgs::PoseStamped>(pose_channel,1);

        depth_channel = nh_.resolveName("lsd_slam/depth");
	depth_publisher = nh_.advertise<sensor_msgs::Image>(depth_channel,1);
        
        depth_gt_channel = nh_.resolveName("lsd_slam/depth_gt");
	depth_gt_publisher = nh_.advertise<sensor_msgs::Image>(depth_gt_channel,1);
        
        depth_corr_channel = nh_.resolveName("lsd_slam/depth_corrected");
	depth_corr_publisher = nh_.advertise<sensor_msgs::Image>(depth_corr_channel,1);
        //image_transport::ImageTransport it(nh_);
        //depth_publisher= it.advertise("camera/image", 1);
        
        keyFrame_channel = nh_.resolveName("lsd_slam/depth_kf");;
	keyFrame_publisher  = nh_.advertise<sensor_msgs::Image>(keyFrame_channel,1);
 
        grayImage_channel = nh_.resolveName("/camera/rgb/image_raw");;
	grayImage_publisher  = nh_.advertise<sensor_msgs::Image>(grayImage_channel,1);
        
        
        depth_var_channel = nh_.resolveName("lsd_slam/depth_var");;
        depth_var_publisher = nh_.advertise<sensor_msgs::Image>(depth_var_channel,1);
        
        scale_channel = nh_.resolveName("lsd_slam/scale");; 
        scale_publisher = nh_.advertise<std_msgs::Float32>(scale_channel,1);
        

	publishLvl=0;
}

ROSOutput3DWrapper::~ROSOutput3DWrapper()
{
}


void ROSOutput3DWrapper::publishKeyframe(Frame* f,float* gtDepth_array,float* curentDepth_array, const float* curImage, SE3 camToKeyframe,float* correctedDepth,float* correctedDepthVar, Frame* curFame)
{
	lsd_slam_viewer::keyframeMsg fMsg;

        std::cout <<"pyub "<<std::endl;
	boost::shared_lock<boost::shared_mutex> lock = f->getActiveLock();

	fMsg.id = f->id();
	fMsg.time = f->timestamp();
	fMsg.isKeyframe = true;

	int w = f->width(publishLvl);
	int h = f->height(publishLvl);

	memcpy(fMsg.camToWorld.data(),f->getScaledCamToWorld().cast<float>().data(),sizeof(float)*7);
	fMsg.fx = f->fx(publishLvl);
	fMsg.fy = f->fy(publishLvl);
	fMsg.cx = f->cx(publishLvl);
	fMsg.cy = f->cy(publishLvl);
	fMsg.width = w;
	fMsg.height = h;


	fMsg.pointcloud.resize(w*h*sizeof(InputPointDense));

	InputPointDense* pc = (InputPointDense*)fMsg.pointcloud.data();

	const float* idepth = f->idepth(publishLvl);
	const float* idepthVar = f->idepthVar(publishLvl);
	const float* color = f->image(publishLvl);
        
        float *keyFrameDepth_array = new float[h*w];
        float *corrDepth_array = new float[h*w];
        unsigned char *gray_img = new unsigned char[h*w];
        float* depth_var_array = new float[h*w];
        float* gtDepth_array_local = new float[h*w];
	for(int idx=0;idx < w*h; idx++)
	{
                keyFrameDepth_array[idx] = idepth[idx];//correctedDepthVar[idx];//
                gray_img[idx] = (unsigned char) color[idx];//curImage[idx];
                depth_var_array[idx] = idepthVar[idx];
                gtDepth_array_local[idx] = gtDepth_array[idx];
                
                /*
                if(std::isnan(gtDepth_array[idx]) || gtDepth_array[idx] == 0){
                    corrDepth_array[idx]= keyFrameDepth_array[idx];
                }else{
                    corrDepth_array[idx]= gtDepth_array[idx];
                }
                 */
                
		pc[idx].idepth = idepth[idx];
		pc[idx].idepth_var = idepthVar[idx];
		pc[idx].color[0] = color[idx];
		pc[idx].color[1] = color[idx];
		pc[idx].color[2] = color[idx];
		pc[idx].color[3] = color[idx];
	}
        float scale = f->getScaledCamToWorld().scale();
	keyframe_publisher.publish(fMsg);
        
        cv::Mat keyFramDepth = cv::Mat(h,w,CV_32FC1,keyFrameDepth_array);//);
        cv::Mat curentDepth ;//= cv::Mat(h,w,CV_32FC1,curentDepth_array);//lsdDepth_array);
                //cv::Mat(2,&sizes[0],CV_32FC1,&idepth);
        cv::Mat gtDepth = cv::Mat(h,w,CV_32FC1,gtDepth_array_local);
        cv::Mat corrDepth ;//= cv::Mat(h,w,CV_32FC1,correctedDepth);
        
        cv::Mat grayImage = cv::Mat(h,w,CV_8UC1, gray_img);
        
        cv::Mat depth_var = cv::Mat(h,w,CV_32FC1,depth_var_array);
            
        std_msgs::Float32  scaleMsg;
        scaleMsg.data = scale;
        scale_publisher.publish(scaleMsg);
        publishDepth2(curentDepth, gtDepth, corrDepth,keyFramDepth, grayImage, depth_var);
        
        //publishTrackedFrame(curentFrame, NULL);
        /**/
        
        if (1 == 1){
         Sophus::SE3f relativeRefrance = se3FromSim3(f->getScaledCamToWorld()).inverse().cast<float>();// camToKeyframe.inverse().cast<float>();
	  Sophus::SE3f referanceToKeyframe = se3FromSim3(curFame->getScaledCamToWorld()).inverse().cast<float>();
         Sophus::SE3f referenceToFrame =  referanceToKeyframe ;//relativeRefrance *
	//Eigen::Matrix3f rotMat = referenceToFrame.rotationMatrix();
	//Eigen::Vector3f transVec = referenceToFrame.translation();
        geometry_msgs::PoseStamped pMsg;

	pMsg.pose.position.x = referenceToFrame.translation()[0];
	pMsg.pose.position.y = referenceToFrame.translation()[1];
	pMsg.pose.position.z = referenceToFrame.translation()[2];
	pMsg.pose.orientation.x = referenceToFrame.so3().unit_quaternion().x();
	pMsg.pose.orientation.y = referenceToFrame.so3().unit_quaternion().y();
	pMsg.pose.orientation.z = referenceToFrame.so3().unit_quaternion().z();
	pMsg.pose.orientation.w = referenceToFrame.so3().unit_quaternion().w();

	if (pMsg.pose.orientation.w < 0)
	{
		pMsg.pose.orientation.x *= -1;
		pMsg.pose.orientation.y *= -1;
		pMsg.pose.orientation.z *= -1;
		pMsg.pose.orientation.w *= -1;
	}

	//pMsg.header.stamp = ros::Time(kf->timestamp());
	pMsg.header.frame_id = "world";
	pose_publisher.publish(pMsg);
        }
}

void ROSOutput3DWrapper::publishTrackedFrame(Frame* kf,float* gtDepth_array)
{
    //if(gtDepth_array == NULL)
    if(0==1)
        {
	lsd_slam_viewer::keyframeMsg fMsg;


	fMsg.id = kf->id();
	fMsg.time = kf->timestamp();
	fMsg.isKeyframe = false;


	memcpy(fMsg.camToWorld.data(),kf->getScaledCamToWorld().cast<float>().data(),sizeof(float)*7);
	fMsg.fx = kf->fx(publishLvl);
	fMsg.fy = kf->fy(publishLvl);
	fMsg.cx = kf->cx(publishLvl);
	fMsg.cy = kf->cy(publishLvl);
	fMsg.width = kf->width(publishLvl);
	fMsg.height = kf->height(publishLvl);

	fMsg.pointcloud.clear();

	liveframe_publisher.publish(fMsg);


	SE3 camToWorld = se3FromSim3(kf->getScaledCamToWorld());

	geometry_msgs::PoseStamped pMsg;

	pMsg.pose.position.x = camToWorld.translation()[0];
	pMsg.pose.position.y = camToWorld.translation()[1];
	pMsg.pose.position.z = camToWorld.translation()[2];
	pMsg.pose.orientation.x = camToWorld.so3().unit_quaternion().x();
	pMsg.pose.orientation.y = camToWorld.so3().unit_quaternion().y();
	pMsg.pose.orientation.z = camToWorld.so3().unit_quaternion().z();
	pMsg.pose.orientation.w = camToWorld.so3().unit_quaternion().w();

	if (pMsg.pose.orientation.w < 0)
	{
		pMsg.pose.orientation.x *= -1;
		pMsg.pose.orientation.y *= -1;
		pMsg.pose.orientation.z *= -1;
		pMsg.pose.orientation.w *= -1;
	}

	pMsg.header.stamp = ros::Time(kf->timestamp());
	pMsg.header.frame_id = "world";
	pose_publisher.publish(pMsg);
        
        }
       
        /*
        int w = kf->width(publishLvl);
	int h = kf->height(publishLvl);
        
        const float* idepth = kf->idepth(publishLvl);
	const float* idepthVar = kf->idepthVar(publishLvl);
	const float* color = kf->image(publishLvl);
        
        float *data = new float[h*w];
	for(int idx=0;idx < w*h; idx++)
	{
            data[idx] = idepth[idx];
        }
        
        cv::Mat depth2 = cv::Mat(h,w,CV_32FC1,data);
                //cv::Mat(2,&sizes[0],CV_32FC1,&idepth);
       
        cv::Mat gtDepth = cv::Mat(h,w,CV_32FC1,gtDepth_array);
        publishDepth2(depth2,gtDepth);
         */
        
}



void ROSOutput3DWrapper::publishKeyframeGraph(KeyFrameGraph* graph)
{
    if(0==1)
        {
	lsd_slam_viewer::keyframeGraphMsg gMsg;

	graph->edgesListsMutex.lock();
	gMsg.numConstraints = graph->edgesAll.size();
	gMsg.constraintsData.resize(gMsg.numConstraints * sizeof(GraphConstraint));
	GraphConstraint* constraintData = (GraphConstraint*)gMsg.constraintsData.data();
	for(unsigned int i=0;i<graph->edgesAll.size();i++)
	{
		constraintData[i].from = graph->edgesAll[i]->firstFrame->id();
		constraintData[i].to = graph->edgesAll[i]->secondFrame->id();
		Sophus::Vector7d err = graph->edgesAll[i]->edge->error();
		constraintData[i].err = sqrt(err.dot(err));
	}
	graph->edgesListsMutex.unlock();

	graph->keyframesAllMutex.lock_shared();
	gMsg.numFrames = graph->keyframesAll.size();
	gMsg.frameData.resize(gMsg.numFrames * sizeof(GraphFramePose));
	GraphFramePose* framePoseData = (GraphFramePose*)gMsg.frameData.data();
	for(unsigned int i=0;i<graph->keyframesAll.size();i++)
	{
		framePoseData[i].id = graph->keyframesAll[i]->id();
		memcpy(framePoseData[i].camToWorld, graph->keyframesAll[i]->getScaledCamToWorld().cast<float>().data(),sizeof(float)*7);
	}
	graph->keyframesAllMutex.unlock_shared();

	graph_publisher.publish(gMsg);
    }
}

void ROSOutput3DWrapper::publishTrajectory(std::vector<Eigen::Matrix<float, 3, 1>> trajectory, std::string identifier)
{
	// unimplemented ... do i need it?
}

void ROSOutput3DWrapper::publishTrajectoryIncrement(Eigen::Matrix<float, 3, 1> pt, std::string identifier)
{
	// unimplemented ... do i need it?
}

void ROSOutput3DWrapper::publishDebugInfo(Eigen::Matrix<float, 20, 1> data)
{
    if(0==1)
        {
	std_msgs::Float32MultiArray msg;
	for(int i=0;i<20;i++)
		msg.data.push_back((float)(data[i]));

	debugInfo_publisher.publish(msg);
    }
}

void ROSOutput3DWrapper::publishDepth(cv::Mat depth){
     /*
     cv::namedWindow( "Display window", cv::WINDOW_AUTOSIZE );// Create a window for display.
     cv::imshow( "Display window", depth );                   // Show our image inside it.
     //cv::waitKey(0); 
     std::cout <<"depth size " << depth.rows <<" "<< depth.cols<<" "<<std::endl;
     */        
    //sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "32FC1", depth).toImageMsg();
    //depth_publisher.publish(msg);
}

void ROSOutput3DWrapper::publishDepth2(cv::Mat lsdDepth,cv::Mat gtDepth, cv::Mat corrDepth , cv::Mat keyFramDepth, cv::Mat grayImage, cv::Mat depth_var){
     /*
     cv::namedWindow( "Display window", cv::WINDOW_AUTOSIZE );// Create a window for display.
     cv::imshow( "Display window", depth );                   // Show our image inside it.
     //cv::waitKey(0); 
     std::cout <<"depth size " << depth.rows <<" "<< depth.cols<<" "<<std::endl;
     */        
    //sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "32FC1", lsdDepth).toImageMsg();
    //depth_publisher.publish(msg);
    
    sensor_msgs::ImagePtr msg_gt = cv_bridge::CvImage(std_msgs::Header(), "32FC1", gtDepth).toImageMsg();
    depth_gt_publisher.publish(msg_gt);
    
    sensor_msgs::ImagePtr msg_kf = cv_bridge::CvImage(std_msgs::Header(), "32FC1", keyFramDepth).toImageMsg();
    keyFrame_publisher.publish(msg_kf);
    
    sensor_msgs::ImagePtr msg_im = cv_bridge::CvImage(std_msgs::Header(), sensor_msgs::image_encodings::MONO8, grayImage).toImageMsg();
    grayImage_publisher.publish(msg_im);
   
    /*
    */
    //sensor_msgs::ImagePtr msg_corr = cv_bridge::CvImage(std_msgs::Header(), "32FC1", corrDepth).toImageMsg();
    //depth_corr_publisher.publish(msg_corr);
    
    
    sensor_msgs::ImagePtr msg_var = cv_bridge::CvImage(std_msgs::Header(), "32FC1", depth_var).toImageMsg();
    depth_var_publisher.publish(msg_var);
     
    
    std::cout << "Writing to bag  "<<std::endl;
  
}
}
