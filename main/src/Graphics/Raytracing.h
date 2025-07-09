/** ***********************************************************************************************
* @class        Raytracing
* @brief        Softrenderer for animation and image generation
*
* @author       Gerstmayr Johannes
* @date         2025-05-23 (generated)
*
* @copyright    This file is part of Exudyn. Exudyn is free software: you can redistribute it and/or modify it under the terms of the Exudyn license. See "LICENSE.txt" for more details.
* @note         Bug reports, support and further information:
                - email: johannes.gerstmayr@uibk.ac.at
                - weblink: https://github.com/jgerstmayr/EXUDYN
                
************************************************************************************************ */
#ifndef RAYTRACING__H
#define RAYTRACING__H

#include "Linalg/BasicLinalg.h"
#include "Linalg/Geometry.h"
#include "Linalg/SearchTree.h"

//#include "Linalg/RigidBodyMath.h"


#ifdef USE_GLFW_GRAPHICS

class GLTriangle;
class GlfwRenderer;
class VisualizationSettings;

typedef float RTfloat; //for raytracing
typedef SlimVectorBase<RTfloat, 3> RTVector3D;
typedef SlimVectorBase<RTfloat, 4> RTVector4D;
typedef ConstSizeMatrixBase<RTfloat, 9> RTMatrix3D;
typedef ConstSizeMatrixBase<RTfloat, 16> RTMatrix4D;

inline RTVector3D ColorRGBA2RGB(const RTVector4D& color4)
{
	return RTVector3D({ color4[0], color4[1], color4[2] });
}

typedef VSettingsMaterial GLMaterial;


class RaytracingLight
{
public:
	//opengl parameters:
	RTVector4D position; //light pos/dir according to openGL
	RTfloat ambient; 
	RTfloat diffuse;
	RTfloat specular;
	RTfloat constantAttenuation;
	RTfloat linearAttenuation;
	RTfloat quadraticAttenuation;
	bool active;
};

class RaytracingSettings
{
public:
	//Raytracing settings:
	Index maxReflectionDepth;
	Index maxTransparencyDepth;
	Index numberOfThreads;
	bool verbose;
	bool keepWindowActive;  //!< flag to call pollEvents to avoid rendering timeouts; may lead to instability in regular use cases, but allows very long rendering (>>5 seconds)
	Index imageSizeFactor;	// subsampling; faster rendering
	RTfloat zBiasLines;		//!< slight bias to make lines visible over coplanar geometry
	RTfloat zOffsetCamera;	//!< allows to go into scene
	//unchanged for now:
	RTfloat minZ;			//!< for triangles and reflections, etc.
	RTVector3D backgroundColorReflections; //!< for refraction
	RTVector3D ambientLightColor;
	RTfloat globalFogDensity;
	RTVector3D globalFogColor;
	RTfloat globalMaxAlpha; //!< for global transparency
	//std::array<GLMaterial, 10> materials; //moved to MainSystemContainer.renderer.materials
	std::vector<VSettingsMaterial>* materials;
	Index tilesPerThread;


	//from visualizationSettings.openGL or general:
	bool useShadow;
	RTfloat lightRadius;
	Index lightRadiusVariations;
	Index AAfactor;			// 2x supersampling; higher quality
	bool showLines;
	RTfloat lineWidth;
	Index imageWidth;
	Index imageHeight;
	bool useGradientBackground;
	RTfloat perspectiveFactor;
	RTVector3D backgroundColor;			//!< for zero-hits; could be white or black, ...
	RTVector3D backgroundColorBottom;	//!< for gradient background, bottom color
	bool useClippingPlane;
	RTVector3D clippingPlaneNormal;
	RTVector3D clippingPlanePoint;
	RTfloat clippingPlaneDistance;
	RTVector3D clippingPlaneColor; 
	Index useClippingPlaneColor; //!< 0=no, 1=clippingPlaneColor, 2=object inner color

	//set during initialization of a single scene:
	RTfloat tolParallel;
	RTfloat tolParallel2;
	RTfloat searchTreeBoxMaxRadius;
	Index numberOfThreadsUsed;
	RTfloat zoom, ratio; //screen zoom and ratio

	Matrix4DF modelView;	//!< updated by caller
	Matrix3DF rotationView;	//!< rotation matrix of modelView; updated by caller
	RTVector3D translationView; //!< translation of modelView; updated by caller


	std::array<RaytracingLight,4> lights;
	static constexpr Index materialOffset = 1000;
	static const Index RTcolorDepth = 4;

	SearchTreeBase<RTfloat> searchTree;
	bool hasSearchTree;
	static const Index maxNThreads = 256;
	mutable ArrayIndex tempTrigIndices[maxNThreads];
	mutable ArrayIndex tempSearchTreeBins[maxNThreads];

	//! convert existing color into material index, alpha and color
	inline void ColorRGBA2MaterialColor(Float4& color, Index& materialIndex, RTfloat& alpha) const
	{
		materialIndex = (Index)color[3] >= materialOffset ? EXUstd::Minimum((Index)color[3]- materialOffset, (Index)materials->size() - 1) : 0;
		alpha = (Index)color[3] >= materialOffset ? (*materials)[materialIndex].alpha : EXUstd::Clamp(color[3]);
		if (color[0] < 0.f) //in case of mixture of colors and material base colors, this will lead to artifacts ...
		{
			color[0] = (*materials)[materialIndex].baseColor[0];
			color[1] = (*materials)[materialIndex].baseColor[1];
			color[2] = (*materials)[materialIndex].baseColor[2];
		}
		alpha = EXUstd::Minimum(alpha, globalMaxAlpha); //for global transparency
		//==> now color is a RGBA color with no material index!
	}


	//! two default lights
	void Initialize();

	RaytracingSettings()
	{
		materials = NULL;
		verbose = false;
		keepWindowActive = false;
		tilesPerThread = 12;
		numberOfThreads = 1;
		numberOfThreadsUsed = 1;
		zOffsetCamera = -0.01f; //always stay away a little bit from max scene

		showLines = true;
		useShadow = true;
		lightRadius = 0.f;
		lightRadiusVariations = 1;
		useGradientBackground = false;
		AAfactor = 1; 
		imageSizeFactor = 1;
		perspectiveFactor = 0;

		hasSearchTree = false;
		maxReflectionDepth = 0;
		maxTransparencyDepth = 0;
		globalMaxAlpha = 0;

		lineWidth = 2.f;
		minZ = 0.0f;

		zBiasLines = 1e-3f; //should be relative to sceen size
		tolParallel = 1e-7f;
		tolParallel2 = tolParallel * tolParallel;
		searchTreeBoxMaxRadius = 1.f; //set when searchtree exists

	}
};

class IntersectionResult 
{
public:
	bool hit;
	bool hitFromBack;
	RTfloat distance;
	RTfloat u, v;
	RTVector3D normal;
	RTVector4D color;
	Index triangleIndex;
	Index materialIndex; //from RGBA color
	RTfloat alpha;		 //from RGBA color or material.alpha

	inline void Init()
	{
		hit = false;
		distance = std::numeric_limits<RTfloat>::max();
	}
};


class Raytracer
{
private:
	RaytracingSettings RTS;

public:
	Raytracer() {}

	unsigned char FloatColorToByte(RTfloat x) { return static_cast<unsigned char>((Index)(EXUstd::Clamp(x) * 255.0f)); }

	//! Perform ray-triangle intersection test
	inline void IntersectRayWithTriangle(const RTVector3D& rayOrigin, const RTVector3D& rayDirection,
		const GLTriangle& triangle, const RaytracingSettings& RTS, IntersectionResult& result);

	//! check intersection with list of triangles; 
	//! write information in closestHit structure; use minDistanceTriangle as lower limit because of clipping planes
	void IntersectRayWithTriangles(const RTVector3D& rayOrigin, const RTVector3D& rayDirection, RTfloat rayLength,
		const ResizableArray<GLTriangle>& triangles, const RaytracingSettings& RTS, IntersectionResult& closestHit,
		bool returnOnSingleHit = false, RTfloat minDistanceTriangle = std::numeric_limits<RTfloat>::lowest());

	//! check if point is in shadow with given light
	bool IsInShadow(const RTVector3D& point, const RTVector3D& lightDirection, const ResizableArray<GLTriangle>& triangles, 
		const RaytracingSettings& RTS, RTfloat maxDistance);

	//refraction for transparency
	RTVector3D Refract(const RTVector3D& I, const RTVector3D& N, RTfloat eta);

	//! compute one pixel color from Raytracing
	RTVector3D ComputePixelColor(const RTVector3D& rayOrigin, const RTVector3D& rayDirection, const ResizableArray<GLTriangle>& triangles,
		const RaytracingSettings& RTS, RTfloat& zDepth, Index reflectionDepth = 0, Index transparencyDepth = 0, 
		const RTVector3D& rayOriginNormalized=RTVector3D({0.f,0.f,0.f}));

	void RasterizeLine(const RTVector3D& s0, const RTVector3D& s1, const RTVector4D& color0, const RTVector4D& color1,
		ResizableArray<RTfloat>& zdepths, ResizableArray<unsigned char>& pixelsRGBA, const RaytracingSettings& RTS);

	//! draw a RGBA 8-bit image with given resolution
	void DrawImageRGBA(const ResizableArray<unsigned char>& pixelsRGBA, Index imageWidth, Index imageHeight, Index openGLwindowWidth, Index openGLwindowHeight);
	
	//! use most of openGL settings also for RTS
	void CopyVisSettings2RTS();

	//! software renderer used instead of OpenGL renderer
	void SoftwareRenderer();


};









#endif //USE_GLFW_GRAPHICS
#endif //include once
