/** ***********************************************************************************************
* @brief        Implementation of GlfwClient
*
* @author       Gerstmayr Johannes
* @date         2019-05-24 (generated)
*
* @copyright    This file is part of Exudyn. Exudyn is free software: you can redistribute it and/or modify it under the terms of the Exudyn license. See "LICENSE.txt" for more details.
* @note         Bug reports, support and further information:
                - email: johannes.gerstmayr@uibk.ac.at
                - weblink: https://github.com/jgerstmayr/EXUDYN
                
************************************************************************************************ */

#include "Graphics/GlfwClient.h"
#include "Graphics/Raytracing.h"
#include "Linalg/SearchTree.h"

#include "Utilities/Parallel.h" //include after 

#ifdef USE_GLFW_GRAPHICS
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
extern GlfwRenderer glfwRenderer;

bool rendererDebug = false;
Index nHits = 0;
Index warned = 10000;

// Perform ray-triangle intersection test
void Raytracer::IntersectRayWithTriangle(const RTVector3D& rayOrigin, const RTVector3D& rayDirection, const GLTriangle& triangle, 
	const RaytracingSettings& RTS, IntersectionResult& result)
{
	//if (rendererDebug) { nHits++; }
	result.Init();

	const RTVector3D& v0 = triangle.points[0];
	const RTVector3D& v1 = triangle.points[1];
	const RTVector3D& v2 = triangle.points[2];

	RTVector3D edge1 = v1 - v0;
	RTVector3D edge2 = v2 - v0;
	RTVector3D pVec = rayDirection.CrossProduct(edge2);
	RTfloat det = edge1 * pVec;

	//if (fabs(det) <= RTS.tolParallel) { return; }  // Ray parallel to triangle
	if (det * det <= RTS.tolParallel * RTS.tolParallel) { return; }  // Ray parallel to triangle

	RTVector3D tVec = rayOrigin - v0;
	RTfloat invDet = 1.0f / det; //division is done here, to save time
	RTfloat u = (tVec * pVec) * invDet;
	if ((u < 0.0f) || (u > 1.f)) { return; }

	RTVector3D qVec = tVec.CrossProduct(edge1);
	RTfloat v = (rayDirection * qVec) * invDet;
	if ((v < 0.0f) || (u + v > 1.f) ) { return; }

	RTfloat t = (edge2 * qVec) * invDet;
	if (t < RTS.minZ) { return; }

	// Fill result
	RTfloat w = 1.0f - u - v;
	result.hit = true;
	result.distance = t;
	result.u = u;
	result.v = v;
	result.triangleIndex = -1; // to be filled by caller
	result.normal = triangle.normals[0] * w + triangle.normals[1] * u + triangle.normals[2] * v;
	result.normal.NormalizeSafe();
	result.color = triangle.colors[0] * w + triangle.colors[1] * u + triangle.colors[2] * v;
	result.color[3] = triangle.colors[0][3]; //do not interpolate material index!
	RTS.ColorRGBA2MaterialColor(result.color, result.materialIndex, result.alpha);

	result.hitFromBack = (rayDirection * result.normal) > 0.f;

	if (result.hitFromBack) { result.normal = -result.normal; } //adjust, such that nobody notices ...

	return;
}

//! check if point is in shadow with given light
bool Raytracer::IsInShadow(const RTVector3D& point, const RTVector3D& lightDirection, const ResizableArray<GLTriangle>& triangles, 
	const RaytracingSettings& RTS, RTfloat maxDistance) 
{
	IntersectionResult shadowHit;
	IntersectRayWithTriangles(point, lightDirection, maxDistance, triangles, RTS, shadowHit, true);

	return (shadowHit.hit && shadowHit.distance < maxDistance);
}


//refraction for transparency
RTVector3D Raytracer::Refract(const RTVector3D& I, const RTVector3D& N, RTfloat eta) 
{
	RTfloat cosi = std::clamp(I * N, -1.0f, 1.0f);
	RTfloat etai = 1.0f, etat = eta;
	RTVector3D n = N;
	if (cosi < 0) {
		cosi = -cosi;
	}
	else {
		std::swap(etai, etat);
		n = -N;
	}
	RTfloat etaRatio = etai / etat;
	RTfloat k = 1 - etaRatio * etaRatio * (1 - cosi * cosi);
	if (k < 0) {
		return RTVector3D{ 0, 0, 0 }; // Total internal reflection
	}
	return I * etaRatio + n * (etaRatio * cosi - std::sqrt(k));
}

void Raytracer::IntersectRayWithTriangles(const RTVector3D& rayOrigin, const RTVector3D& rayDirection, RTfloat rayLength,
	const ResizableArray<GLTriangle>& triangles, const RaytracingSettings& RTS, IntersectionResult& closestHit,
	bool returnOnSingleHit, RTfloat minDistanceTriangle)
{
	Index maxInd = triangles.NumberOfItems();
	bool useSearchTree = RTS.hasSearchTree;

	Index threadID = RTS.numberOfThreads==1 ? 0 : ExuThreading::TaskManager::GetThreadId();
	if (useSearchTree)
	{
		RTS.searchTree.GetBinsCrossedByLine(rayOrigin, rayDirection * rayLength, 
			RTS.tempSearchTreeBins[threadID]);
		RTS.searchTree.GetItemsFromBins(RTS.tempSearchTreeBins[threadID], RTS.tempTrigIndices[threadID]);
		maxInd = RTS.tempTrigIndices[threadID].NumberOfItems();

	}

	IntersectionResult hit;
	closestHit.Init();
	for (Index i = 0; i < maxInd; ++i)
	{
		Index iInd = useSearchTree ? RTS.tempTrigIndices[threadID][i] : i;
		IntersectRayWithTriangle(rayOrigin, rayDirection, triangles[iInd], RTS, hit);
		if (hit.hit && hit.distance < closestHit.distance && 
			(hit.distance /*+ RTS.tolParallel*100.f*/ >= minDistanceTriangle) )
		{
			hit.triangleIndex = i;
			closestHit = hit;
			if (returnOnSingleHit) { return; }
		}
	}
}
//for screen pixel or for reflection/refraction: compute pixel color of given ray origin and direction
RTVector3D Raytracer::ComputePixelColor(const RTVector3D& rayOrigin, const RTVector3D& rayDirection,
	const ResizableArray<GLTriangle>& triangles, const RaytracingSettings& RTS, RTfloat& zDepth,
	Index reflectionDepth, Index transparencyDepth, const RTVector3D& rayOriginNormalized)
{
	IntersectionResult closestHit;

	if (RTS.useClippingPlane && reflectionDepth == 0 && transparencyDepth == 0)
	{
		RTfloat minDistanceTriangle = std::numeric_limits<RTfloat>::lowest();
		bool hitFront;

		RTfloat distance;

		hitFront = RTS.clippingPlaneNormal * rayDirection < 0.f;
		if (EGeometry::LinePlaneIntersection(RTS.clippingPlanePoint, -RTS.clippingPlaneNormal,
			rayOrigin, rayDirection, distance))
		{
			minDistanceTriangle = hitFront ? distance : std::numeric_limits<RTfloat>::lowest();
			IntersectRayWithTriangles(rayOrigin, rayDirection, RTS.searchTreeBoxMaxRadius * 4, triangles, RTS,
				closestHit, false, minDistanceTriangle);
			if (hitFront)
			{
				if (closestHit.hit && closestHit.hitFromBack) //fails for wrong normals!
				{
					if (RTS.useClippingPlaneColor == 1)
					{
						return  RTS.clippingPlaneColor;
					}
					else if (RTS.useClippingPlaneColor == 2)
					{
						//special, but this doesnt work well for objects in objects
						return 0.75f * ColorRGBA2RGB(closestHit.color); 
					}
					else 
					{ 
						//do regular rendering
					}
				}
			}
			else
			{
				if (closestHit.distance > distance)
				{
					closestHit.hit = false; //to use background
				}
			}
		}
		else //orthogonal
		{
			if (RTS.clippingPlaneNormal * (rayOrigin - RTS.clippingPlanePoint) > 0)
			{ 
				closestHit.hit = false; 
			}
			else
			{
				IntersectRayWithTriangles(rayOrigin, rayDirection, RTS.searchTreeBoxMaxRadius * 4, triangles, RTS,
					closestHit);
			}
		}
		//minDistanceTriangle = 5;
	}
	else
	{
		IntersectRayWithTriangles(rayOrigin, rayDirection, RTS.searchTreeBoxMaxRadius * 4, triangles, RTS,
			closestHit);
	}

	if (!closestHit.hit)
	{
		if (reflectionDepth == 0)
		{
			zDepth = rayDirection[2] * std::numeric_limits<RTfloat>::max();

			if (!RTS.useGradientBackground)
			{
				return RTS.backgroundColor;
			}
			else
			{
				return RTS.backgroundColor * 0.5f * (1.f + rayOriginNormalized[1]) +
					RTS.backgroundColorBottom * 0.5f * (1.f - rayOriginNormalized[1]);
				//return RTVector3D({ 1,0,0 }) *0.5f * (1.f + rayOriginNormalized[1]) +
				//	RTVector3D({ 0,0,1 }) * 0.5f * (1.f - rayOriginNormalized[1]);
			}
		}
		else { return RTS.backgroundColorReflections; }
	}
	else
	{
		zDepth = rayOrigin[2] + rayDirection[2] * closestHit.distance;
	}

	const GLMaterial& material = (*RTS.materials)[closestHit.materialIndex];

	const GLTriangle& triangle = triangles[closestHit.triangleIndex];
	RTVector3D hitPoint = rayOrigin + rayDirection * closestHit.distance;

	// Ambient
	RTVector3D shadedColor;
	EXUmath::MultVectorComponents(RTS.ambientLightColor,
		RTVector3D({ closestHit.color[0], closestHit.color[1], closestHit.color[2] }),
		shadedColor);
	shadedColor += material.emission;
	RTfloat otherZ; //unused, dummy

	for (const RaytracingLight& light : RTS.lights) {
		if (!light.active) continue;

		//15 variations: center, vertices and face midpoints of cube
		Index nLightPoints = (RTS.useShadow
			&& light.position[3] != 0.f
			&& RTS.lightRadius > 0.f && RTS.lightRadiusVariations > 1
			&& reflectionDepth == 0 && transparencyDepth == 0) ? EXUstd::Minimum(64,RTS.lightRadiusVariations) : 1;

		for (Index np = 0; np < nLightPoints; np++)
		{
			RTVector3D lightDir;
			RTfloat lightDistance = std::numeric_limits<RTfloat>::max();
			RTVector3D lightPos = { light.position[0], light.position[1], light.position[2] };
			GlfwRenderer::TransformVertex(lightPos, RTS.modelView, lightDir); //lights are not rotated; now in rotated config

			//std::cout << "np=" << np << "\n";
			RTfloat factLights = 1.f / (RTfloat)nLightPoints;
			if (light.position[3] == 0.f) //only direction is used!
			{
				lightDir.NormalizeSafe();
				lightDistance = (lightDir - hitPoint).GetL2Norm(); //for attenuation, we still compute distance!
			}
			else //light is used as position
			{
				//option to use several point lights (but systematic, to avoid noise)
				lightDir -= hitPoint; //now lightDir is the direction
				lightDistance = EXUstd::Maximum(1e-5f, lightDir.GetL2Norm());
				lightDir *= 1.f / lightDistance;
				RTVector3D vec(0.f);
				if (nLightPoints > 1)
				{
					RTfloat x = RTS.lightRadius / lightDistance * (RTfloat)sin(EXUstd::pi * 2. * np / nLightPoints);
					RTfloat y = RTS.lightRadius / lightDistance * (RTfloat)cos(EXUstd::pi * 2. * np / nLightPoints);
					RTVector3D basisN1, basisN2;
					EXUmath::ComputeOrthogonalBasisVectors(lightDir, basisN1, basisN2);

					lightDir += x * basisN1 + y * basisN2; //lightDir not normalized any more, but should be ok
				}
				//if (np > 0)
				//{
				//	if (np < 9) //1..8
				//	{
				//		Index np0 = np - 1;
				//		vec = RTVector3D({ (RTfloat)(np0 % 2) - 0.5f, (RTfloat)((np0 / 2) % 2) - 0.5f, (RTfloat)((np0 / 4) % 2) - 0.5f });
				//		lightDir += (2.f / 1.7320508f * RTS.lightRadius) * vec;
				//	}
				//	else //9..14
				//	{
				//		Index np0 = np - 9;
				//		RTfloat x = np0 < 2 ? -1.f + 2.f*(RTfloat)np0 : 0.f;
				//		RTfloat y = (np0 >= 2 && np0 < 4) ? -1.f + 2.f * (RTfloat)(np0 - 2) : 0.f;
				//		RTfloat z = np0 >= 4 ? -1.f + 2.f * (RTfloat)(np0 - 4) : 0.f;

				//		lightDir +=  RTS.lightRadius * RTVector3D({ x,y,z });
				//	}
				//}
				//lightDistance = lightDir.GetL2Norm();
				//if (lightDistance != 0) { lightDir *= 1.f / lightDistance; }
			}
			//bool isInShadow = RTS.useShadow ? IsInShadow(hitPoint + lightDir * 0.01f, lightDir, triangles, RTS, lightDistance) : false;
			if (RTS.useShadow)
			{
				if (IsInShadow(hitPoint + lightDir * 0.01f, lightDir, triangles, RTS, lightDistance)) {
					continue;
				}
			}

			// Diffuse
			RTfloat diffuseIntensity = std::max(0.0f, closestHit.normal * lightDir);
			RTfloat attenuation = 1.0f / (
				light.constantAttenuation +
				light.linearAttenuation * lightDistance +
				light.quadraticAttenuation * lightDistance * lightDistance
				);

			RTVector3D lightContribution = {
				closestHit.color[0] * light.diffuse * diffuseIntensity * attenuation,
				closestHit.color[1] * light.diffuse * diffuseIntensity * attenuation,
				closestHit.color[2] * light.diffuse * diffuseIntensity * attenuation
			};

			shadedColor += factLights * lightContribution;


			if (np == 0) //only for center of light
			{
				// Specular (Phong)
				RTVector3D viewDir = -rayDirection;
				RTVector3D halfVec = (lightDir + viewDir);
				halfVec.NormalizeSafe();

				RTfloat adjustedShininess = std::max(1.0f, material.shininess);
				//RTfloat adjustedShininess = std::max(1.0f, material.shininess * (1.0f - material.roughness)); //for simplification, roughness is not used for now!
				RTfloat specIntensity = std::pow(std::max(0.0f, closestHit.normal * halfVec), adjustedShininess);
				shadedColor += material.specular * light.specular * specIntensity * attenuation;
			}
		}
	}

	// Reflection (recursive)
	RTVector3D reflectionColor = { 0, 0, 0 };
	if (material.reflectivity > 0 && reflectionDepth < RTS.maxReflectionDepth )
	{
		RTVector3D reflectDir = rayDirection - 2.0f * (rayDirection * closestHit.normal) * closestHit.normal;
		reflectDir.NormalizeSafe();
		reflectionColor = ComputePixelColor(hitPoint + reflectDir * 0.002f, reflectDir, triangles, RTS, otherZ, 
			reflectionDepth + 1,
			EXUstd::Maximum(reflectionDepth, transparencyDepth)); //for now, transparency not used with reflection
	}

	//Transparency/Refraction
	RTVector3D refractionColor({ 0.f,0.f,0.f });
	RTfloat alphaUsed = closestHit.alpha;
	//alphaUsed = 1.f;

	if (alphaUsed < 1.0f && transparencyDepth < RTS.maxTransparencyDepth) {
		RTVector3D refractDir = Refract(rayDirection, closestHit.normal, material.ior);
		refractDir.NormalizeSafe();
		if (refractDir.GetL2NormSquared() > 0.0f)
		{  // No total internal reflection
			refractionColor = ComputePixelColor(hitPoint + refractDir * 0.002f, refractDir,
				triangles, RTS, otherZ, EXUstd::Maximum(reflectionDepth, transparencyDepth), transparencyDepth + 1, 
				rayOriginNormalized); //use original rayOriginNormalized for gradient background in case of pure transparency, if we look through objects (not considering then refraction!)
		}
		else
		{
			refractionColor = RTS.backgroundColor;
		}
	}
	else { alphaUsed = 1.f; } //to keep original color in case that there is no transparency!

	// Combine: reflection + refraction + local shading
	RTVector3D localWithReflection = shadedColor * (1.0f - material.reflectivity) + reflectionColor * material.reflectivity;
	RTVector3D finalColor = localWithReflection * alphaUsed + refractionColor * (1.0f - alphaUsed);

	if (RTS.globalFogDensity > 0.f)
	{
		// distance from camera to hit point
		RTfloat fogDistance = closestHit.distance; (hitPoint - rayOrigin).GetL2Norm();
		fogDistance /= 2.f * RTS.searchTreeBoxMaxRadius; //normalize distance

		// exponential fog factor (more realistic than linear)
		RTfloat fogFactor = std::exp(-RTS.globalFogDensity * fogDistance);
		fogFactor = std::clamp(fogFactor, 0.0f, 1.0f); // ensure in [0, 1]

		// blend final color with fog
		finalColor = finalColor * fogFactor + RTS.globalFogColor * (1.0f - fogFactor);
	}

	return finalColor;
}

void Raytracer::RasterizeLine(const RTVector3D& s0, const RTVector3D& s1, const RTVector4D& color0, const RTVector4D& color1,
							  ResizableArray<RTfloat>& zdepths, ResizableArray<unsigned char>& pixelsRGBA,
							  const RaytracingSettings& RTS)
{
	auto ScreenToModelView = [&](RTfloat x, RTfloat y, RTfloat z) -> RTVector3D {

		RTfloat mx = RTS.zoom * RTS.ratio * (x / (0.5f * RTS.imageWidth) - 1.f);
		RTfloat my = RTS.zoom * (y / (0.5f * RTS.imageHeight) - 1.f);

		return RTVector3D({ mx, my, z });
		};

	RTfloat lineWidth = RTS.lineWidth * RTS.AAfactor;

	RTfloat dx = s1[0] - s0[0];
	RTfloat dy = s1[1] - s0[1];
	RTfloat length = std::sqrt(dx * dx + dy * dy);
	if (length < 1e-6f) return;
	const bool smoothLine = true;

	int steps = static_cast<int>(length * 1.5f); // subpixel steps for smooth line
	int radius = std::max(0, int(std::round(lineWidth / 2.f + 0.1f)));
	for (int i = 0; i <= steps; ++i)
	{
		RTfloat t = (RTfloat)i / (RTfloat)steps;

		RTfloat x = s0[0] + dx * t;
		RTfloat y = s0[1] + dy * t;
		RTfloat z = s0[2] + (s1[2] - s0[2]) * t;
		
		if (RTS.useClippingPlane)
		{
			//check if line is in clipping plane
			//RTVector3D p({ x,y,z });
			if (RTS.clippingPlaneNormal * (ScreenToModelView(x,y,z) - RTS.clippingPlanePoint) > 0)
			{
				continue; //skip point
			}

		}

		RTVector4D color = (1.f - t) * color0 + t * color1; //({ 0,0,0,1 });

		for (int oy = -radius; oy <= radius; ++oy)
		{
			for (int ox = -radius; ox <= radius; ++ox)
			{
				int px = int(std::round(x)) + ox;
				int py = int(std::round(y)) + oy;

				if (px < 0 || px >= RTS.imageWidth || py < 0 || py >= RTS.imageHeight) continue;

				Index pixelIdx = py * RTS.imageWidth + px;
				RTfloat zWithBias = z + RTS.zBiasLines;
				
				if (zWithBias > zdepths[pixelIdx])
				{
					zdepths[pixelIdx] = 1e10;//zWithBias;
					Index iRGBA = pixelIdx * RTS.RTcolorDepth;
					RTfloat alpha = color[3];
					for (int c = 0; c < 3; ++c) // RGB channels
					{
						RTfloat existing = pixelsRGBA[iRGBA + c] / 255.f;
						RTfloat newValue = color[c] * alpha + existing * (1.0f - alpha);
						pixelsRGBA[iRGBA + c] = FloatColorToByte(newValue);
					}
				}
			}
		}
	}
}

void Raytracer::CopyVisSettings2RTS()
{
	VisualizationSettings* visSettings = glfwRenderer.GetVisualizationSettings();

	//copy current material information from visSettings (change of material always updates visSettings!)
	glfwRenderer.GetVisualizationSystemContainer()->CopyMaterialsFromVisualizationSettings();

	//Raytracing settings:
	RTS.maxReflectionDepth = EXUstd::Clamp(visSettings->raytracer.maxReflectionDepth, 0, 32);
	RTS.maxTransparencyDepth = EXUstd::Clamp(visSettings->raytracer.maxTransparencyDepth, 0, 32);
	RTS.keepWindowActive = visSettings->raytracer.keepWindowActive;
	RTS.backgroundColorReflections = ColorRGBA2RGB(visSettings->raytracer.backgroundColorReflections);
	RTS.ambientLightColor = ColorRGBA2RGB(visSettings->raytracer.ambientLightColor);
	RTS.globalFogDensity = (RTfloat)visSettings->raytracer.globalFogDensity;
	RTS.globalFogColor = ColorRGBA2RGB(visSettings->raytracer.globalFogColor);
	RTS.lightRadius = visSettings->raytracer.lightRadius;
	RTS.lightRadiusVariations = visSettings->raytracer.lightRadiusVariations;

	RTS.verbose = visSettings->raytracer.verbose;
	RTS.numberOfThreads = EXUstd::Clamp(visSettings->raytracer.numberOfThreads, 1, RTS.maxNThreads);
	RTS.tilesPerThread = visSettings->raytracer.tilesPerThread;
	RTS.imageSizeFactor = EXUstd::Clamp(visSettings->raytracer.imageSizeFactor, 1, 16);
	RTS.zOffsetCamera = (RTfloat)visSettings->raytracer.zOffsetCamera;
	//RTS.zBiasLines //at raytracer

	//openGL settings:
	RTS.AAfactor = EXUstd::Clamp(visSettings->openGL.multiSampling,1,4); // 2x supersampling; higher quality
	RTS.showLines = visSettings->openGL.showLines;
	RTS.useClippingPlane = visSettings->openGL.clippingPlaneNormal.GetL2NormSquared() != 0.f;
	//RTS.clippingPlaneNormal = visSettings->openGL.clippingPlaneNormal; //done with transformation later
	RTS.clippingPlaneDistance = visSettings->openGL.clippingPlaneDistance;
	RTS.clippingPlaneColor = ColorRGBA2RGB(visSettings->openGL.clippingPlaneColor);
	RTS.useClippingPlaneColor = (Index)visSettings->openGL.clippingPlaneColor[3];
	RTS.globalMaxAlpha = visSettings->openGL.facesTransparent ? visSettings->openGL.faceTransparencyGlobal : 1.f;

	//visibility settings:
	RTS.lineWidth = visSettings->openGL.lineWidth;
	RTS.zBiasLines = visSettings->openGL.polygonOffset;
	RTS.perspectiveFactor = visSettings->openGL.perspective;
	RTS.useShadow = visSettings->openGL.shadow != 0.f;
	RTS.useGradientBackground = visSettings->general.useGradientBackground;

	RTS.backgroundColor = ColorRGBA2RGB(visSettings->general.backgroundColor);
	RTS.backgroundColorBottom = ColorRGBA2RGB(visSettings->general.backgroundColorBottom);
	
	//lights:
	//Light 0
	RTS.lights[0].active = visSettings->openGL.enableLight0;
	RTS.lights[0].ambient = visSettings->openGL.light0ambient;
	RTS.lights[0].diffuse = visSettings->openGL.light0diffuse;
	RTS.lights[0].specular = visSettings->openGL.light0specular;
	RTS.lights[0].constantAttenuation = visSettings->openGL.light0constantAttenuation;
	RTS.lights[0].linearAttenuation = visSettings->openGL.light0linearAttenuation;
	RTS.lights[0].quadraticAttenuation = visSettings->openGL.light0quadraticAttenuation;
	RTS.lights[0].position = visSettings->openGL.light0position;

	//Light 1
	RTS.lights[1].active = visSettings->openGL.enableLight1;
	RTS.lights[1].ambient = visSettings->openGL.light1ambient;
	RTS.lights[1].diffuse = visSettings->openGL.light1diffuse;
	RTS.lights[1].specular = visSettings->openGL.light1specular;
	RTS.lights[1].constantAttenuation = visSettings->openGL.light1constantAttenuation;
	RTS.lights[1].linearAttenuation = visSettings->openGL.light1linearAttenuation;
	RTS.lights[1].quadraticAttenuation = visSettings->openGL.light1quadraticAttenuation;
	RTS.lights[1].position = visSettings->openGL.light1position;

	//Deactivate other lights
	for (Index i = 2; i < 4; ++i) 
	{
		RTS.lights[i].active = false;
	}

}

//! software renderer used instead of OpenGL renderer
void Raytracer::SoftwareRenderer()
{
	Real tStart = EXUstd::GetTimeInSeconds();
	VisualizationSettings* visSettings = glfwRenderer.GetVisualizationSettings();
	//do this at the very begining, as materials are needed!
	RTS.materials = glfwRenderer.GetVisualizationSystemContainer()->GetGraphicsMaterialList();

	RTS.zoom = glfwRenderer.GetRenderStatePtr()->zoom;

	Index width, height;
	glfwRenderer.GetWindowSize(width, height);
	RTS.ratio = (RTfloat)width;
	if (height != 0) { RTS.ratio = width / (RTfloat)height; }

	//RaytracingSettings RTS;
	RTS.Initialize();
	RTS.modelView = glfwRenderer.GetRenderStatePtr()->modelRotation;
	Matrix3DF rotationMV = EXUmath::Matrix4DtoMatrix3D(RTS.modelView);
	RTVector3D translationMV = glfwRenderer.GetRenderStatePtr()->rotationCenterPoint * rotationMV + glfwRenderer.GetRenderStatePtr()->centerPoint;
	RTS.rotationView = EXUmath::Matrix4DtoMatrix3D(RTS.modelView);
	RTS.translationView = translationMV;

	//add translation to transposed matrix (is used in transposed form)!
	RTS.modelView(3, 0) = -translationMV[0];
	RTS.modelView(3, 1) = -translationMV[1];

	CopyVisSettings2RTS();


	RTS.imageWidth = EXUstd::Maximum(4,(RTS.AAfactor*width) / RTS.imageSizeFactor);
	RTS.imageHeight = EXUstd::Maximum(4, (RTS.AAfactor*height) / RTS.imageSizeFactor);

	static ResizableArray<GLTriangle> triangles; //first transform triangles into view
	triangles.SetNumberOfItems0();
	static ResizableArray<GLLine> lines;		//first transform lines into view
	lines.SetNumberOfItems0();

	//this could be parallelized:
	//transform points and normals and compute bounding box:
	Box3DBase<RTfloat> box3D;
	Index cntWrongNormals = 0;
	static bool warnedWrongNormals = false;
	for (auto data : *glfwRenderer.GetGraphicsDataList())
	{
		for (const GLTriangle& trig : data->glTriangles)
		{
			GLTriangle trigNew(trig);

			for (Index i = 0; i < 3; i++)
			{
				GlfwRenderer::TransformVertex(trig.points[i], RTS.modelView, trigNew.points[i]);
				//trigNew.points[i] = trig.points[i] * RTS.rotationView + RTS.translationView;
				trigNew.normals[i] = trig.normals[i] * RTS.rotationView;

				box3D.Add(trigNew.points[i]);
			}
			Float3 n = EGeometry::ComputeTriangleNormalTemplate<RTfloat, std::array<Float3, 3>>(trig.points);
			if (n * trig.normals[0] < 0.) { cntWrongNormals++; }

			if ((!trig.isFiniteElement && visSettings->openGL.showFaces) ||
				(trig.isFiniteElement && visSettings->openGL.showMeshFaces))
			{
				triangles.Append(trigNew);
			}
			if (RTS.showLines && visSettings->openGL.showFaceEdges)
			{
				for (Index i = 0; i < 3; i++)
				{
					GLLine lineNew;
					lineNew.color1 = visSettings->openGL.faceEdgesColor;
					lineNew.color2 = visSettings->openGL.faceEdgesColor;
					lineNew.point1 = trigNew.points[i];
					lineNew.point2 = trigNew.points[(i + 1) % 3];
					lines.Append(lineNew);
				}
			}
		}
		if (RTS.showLines)
		{
			for (const GLLine& line : data->glLines)
			{
				GLLine lineNew(line);
				GlfwRenderer::TransformVertex(line.point1, RTS.modelView, lineNew.point1);
				GlfwRenderer::TransformVertex(line.point2, RTS.modelView, lineNew.point2);
				lines.Append(lineNew);
			}
		}
	}
	if (cntWrongNormals && !warnedWrongNormals)
	{
		warnedWrongNormals = true;
		GlfwRenderer::PrintDelayed("********\nWARNING: Raytracer: There are "+EXUstd::ToString(cntWrongNormals)+" wrong normals in the geometry\n********");
	}

	if (!box3D.Empty()) //is empty at beginning!
	{

		//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
		// some settings which require bounding box or transformation:
		//anyway, we have some offset in the origin!
		RTS.searchTreeBoxMaxRadius = RTS.searchTree.GetBox().Radius();
		RTS.zBiasLines = (RTfloat)visSettings->raytracer.zBiasLines * 2.f * RTS.searchTreeBoxMaxRadius; //relative to sceen size
		RTS.tolParallel = 1e-7f * 2.f * RTS.searchTreeBoxMaxRadius; //fixed for now
		RTS.tolParallel2 = RTS.tolParallel * RTS.tolParallel;
		//avoid this, as the user may get confused by that: box3D.InflateFactor(1.002f);

		RTS.clippingPlaneNormal = visSettings->openGL.clippingPlaneNormal;
		RTS.clippingPlanePoint = RTS.clippingPlaneDistance * RTS.clippingPlaneNormal;
		RTS.clippingPlaneNormal = RTS.clippingPlaneNormal * RTS.rotationView;
		GlfwRenderer::TransformVertex(RTS.clippingPlanePoint, RTS.modelView, RTS.clippingPlanePoint);
		//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++



		Index factST = EXUstd::Clamp(visSettings->raytracer.searchTreeFactor, 1, 128); //additional factor for testing ...
		Index sx, sy, sz;
		SearchTreeBase<RTfloat>::ComputeSearchTreeBinCounts(triangles.NumberOfItems()*factST, box3D, sx, sy, sz);

		//std::cout << "sizes=" << sx << "," << sy << "," << sz << "\n";

		RTS.searchTree.ResetSearchTree(2*sx,2*sy,sz, box3D);
		RTS.hasSearchTree = true;

		for (Index i = 0; i < triangles.NumberOfItems(); ++i)
		{
			const GLTriangle& triangle = triangles[i];
			Box3DBase<RTfloat> tBox;
			for (Index i = 0; i < 3; i++) { tBox.Add(triangle.points[i]); }

			RTS.searchTree.AddItemTriangle(tBox, i, triangle.points[0], triangle.points[1], triangle.points[2]);
		}



		static ResizableArray<unsigned char> pixelsRGBA;
		pixelsRGBA.SetNumberOfItems(RTS.imageWidth * RTS.imageHeight * RTS.RTcolorDepth);
		static ResizableArray<RTfloat> zDepths;
		zDepths.SetNumberOfItems(RTS.imageWidth * RTS.imageHeight);

		RTVector3D rayDirection({ 0,0,-1 });
		RTfloat zOrigin = RTS.searchTree.GetBox().PMaxZ(); //always outside of box
		zOrigin -= RTS.zOffsetCamera;
		//!convert 
		auto ToModelView = [&](Index x, Index y) -> RTVector3D {
			RTfloat fx = static_cast<RTfloat>(x);
			RTfloat fy = static_cast<RTfloat>(y);

			RTfloat screenX = ((2.f * fx) / RTS.imageWidth - 1.f) * RTS.zoom * RTS.ratio;
			RTfloat screenY = ((2.f * fy) / RTS.imageHeight - 1.f) * RTS.zoom;

			return RTVector3D({ screenX, screenY, zOrigin });
			};

		// Returns a ray origin and direction for a given screen pixel
		// perspectiveFactor = 0 -> orthographic (direction = [0, 0, -1])
		// perspectiveFactor = 1 -> strong perspective (eye at z = maxZ + some margin)
		auto ToPerspectiveRay = [&](Index x, Index y, RTfloat perspectiveFactor,
			RTfloat sceneDepth) -> std::pair<RTVector3D, RTVector3D>
			{
				// Use existing orthographic screen mapping
				RTVector3D rayOrigin = ToModelView(x, y); // screen-space origin at zOrigin
				RTfloat zOrigin = rayOrigin[2];
				// Compute eye (camera) position behind scene based on perspective factor
				RTfloat eyeZ = zOrigin + perspectiveFactor * sceneDepth * 1.5f; // 1.5 is arbitrary margin multiplier

				RTVector3D eyePos = { 0.f, 0.f, eyeZ };

				// Interpolate direction: for 0 -> orthographic, for 1 -> full perspective
				RTVector3D orthoDir = { 0, 0, -1 };
				RTVector3D perspDir = rayOrigin - eyePos;

				RTVector3D rayDir = orthoDir * (1.f - perspectiveFactor) + perspDir * perspectiveFactor;
				rayDir.NormalizeSafe();

				return { rayOrigin, rayDir };
			};

		//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
		RTS.numberOfThreadsUsed = 1;

		if (ExuThreading::TaskManager::IsRunning()) //should not happen; only if FinalizeSolver has not been called in last computation
		{
			if (RTS.verbose) { GlfwRenderer::PrintDelayed("Raytracer: multi-threading: TaskManager already/still running (in solver?); do not use multithreading for solver"); }
			RTS.numberOfThreads = 1;
		}
		RTS.numberOfThreadsUsed = RTS.numberOfThreads;
		if (RTS.numberOfThreads > 1)
		{
			ExuThreading::TaskManager::SetNumThreads(RTS.numberOfThreads);

			ExuThreading::EnterTaskManager(); //this is needed in order that any ParallelFor is executed in parallel during solving

			RTS.numberOfThreadsUsed = ExuThreading::TaskManager::GetNumThreads();

			if (RTS.numberOfThreadsUsed == 1) //this reflects that taskmanager is not running!
			{
				if (RTS.verbose) { GlfwRenderer::PrintDelayed("Raytracer: multi-threading failed"); }
			}
			else
			{
				if (RTS.verbose) { GlfwRenderer::PrintDelayed("Raytracer: Start multi-threading with " +EXUstd::ToString(RTS.numberOfThreadsUsed)+ " threads"); }
			}
		}

		Real tStartRT = EXUstd::GetTimeInSeconds();

		//RTS.perspectiveFactor = 1.f; //then also zoom does not work!
		Index nBoxes = RTS.numberOfThreadsUsed*RTS.tilesPerThread;

		Index taskSplit = nBoxes; //RTS.imageHeight / 10;
		//determine tiling in 2D: find grid dimensions close to square
		Index tilesX = EXUstd::Maximum(1, (Index)std::sqrt(nBoxes));
		while (nBoxes % tilesX != 0) { --tilesX; }
		//tilesX = 1;
		Index tilesY = EXUstd::Maximum(nBoxes / tilesX, 1);

		Index tileWidth = RTS.imageWidth / tilesX;
		Index tileHeight = RTS.imageHeight / tilesY;

		//Real complexity = ((Real)RTS.imageHeight * (Real)RTS.imageWidth * (Real)triangles.NumberOfItems()) / 
		//	EXUstd::Maximum(1., (Real)RTS.searchTree.NumberOfCells()* sqrt((Real)RTS.numberOfThreadsUsed) );

		//if (RTS.verbose) { GlfwRenderer::PrintDelayed("complexity: " + EXUstd::ToString(complexity)); }

		//ExuThreading::ParallelFor(RTS.imageHeight, [this, &ToModelView, &ToPerspectiveRay, &rayOrigin](ParallelSizeType index)
		ExuThreading::ParallelFor(nBoxes, [this, &ToPerspectiveRay,
			&tilesX, &tilesY, &tileWidth, &tileHeight, &tStartRT, &nBoxes](ParallelSizeType index)
			{
				Index threadID = RTS.numberOfThreads == 1 ? 0 : ExuThreading::TaskManager::GetThreadId();
				if (threadID == 0) //main thread
				{
					if (EXUstd::GetTimeInSeconds() - tStartRT > 0.5)
					{
						if (RTS.keepWindowActive)
						{
							glfwPollEvents(); //avoid strange timeouts by catching glfw window callbacks in main thread
							//this also works somehow, but clearly leads to major artifacts: glfwSwapBuffers(GlfwRenderer::GetGlfwWindow());
						}
						if (RTS.verbose) //main thread
						{
							GlfwRenderer::PrintDelayed("\rrendering: " + EXUstd::ToString(std::round((Real)index / (Real)nBoxes * 100.))+"%", false, true);
						}
					}
				}
				Index tx = (Index)index % tilesX;
				Index ty = (Index)index / tilesX;

				// Compute tile bounds
				Index xStart = tx * tileWidth;
				Index xEnd = (tx == tilesX - 1) ? RTS.imageWidth : xStart + tileWidth;

				Index yStart = ty * tileHeight;
				Index yEnd = (ty == tilesY - 1) ? RTS.imageHeight : yStart + tileHeight;

				Real val = 0;
				RTVector3D color;
				for (Index y = yStart; y < yEnd; ++y)
				{
					for (Index x = xStart; x < xEnd; ++x)
					{
						//without boxes:
						//Index y = (Index)index;
						//for (Index y = 0; y < RTS.imageHeight; ++y)
						//{
							//for (Index x = 0; x < RTS.imageWidth; ++x)
							//{
						RTVector3D rayOrigin0 = RTVector3D({ ((2.f * x) / RTS.imageWidth - 1.f),
							((2.f * y) / RTS.imageHeight - 1.f), 0.f });

						Index iPix = (y * RTS.imageWidth + x);
						Index i = iPix * RTS.RTcolorDepth;
						auto [rayOrigin, rayDir] = ToPerspectiveRay(x, y, RTS.perspectiveFactor,
							RTS.searchTree.GetBox().SizeZ());

						color = ComputePixelColor(rayOrigin, rayDir, triangles, RTS, zDepths[iPix],0,0, rayOrigin0);

						for (Index j = 0; j < 3; j++)
						{
							pixelsRGBA[i + j] = FloatColorToByte(color[j]);
						}
						pixelsRGBA[i + 3] = 255;
					}
				}
				//DELETE:pixelsRGBA[(yStart * RTS.imageWidth + xStart) * RTS.RTcolorDepth + 3] = (unsigned char)1/val;
			}, taskSplit);

		if (RTS.numberOfThreadsUsed != 1 && //do not stop threads, if running in solver thread!
			ExuThreading::TaskManager::IsRunning()) 
		{
			if (RTS.verbose) { GlfwRenderer::PrintDelayed("Stop multi-threading"); }
			ExuThreading::ExitTaskManager(1); // output.numberOfThreadsUsed);
			ExuThreading::TaskManager::SetNumThreads(1); //for next computation, if it is going to be serial
			RTS.numberOfThreadsUsed = 1;
		}


		//std::cout << "zdepths=" << zDepths << "\n";
		auto ToScreen = [&](const RTVector3D& p) -> RTVector3D {
			RTfloat sx = (p[0] / (RTS.zoom * RTS.ratio) + 1.f) * 0.5f * RTS.imageWidth;
			RTfloat sy = (p[1] / RTS.zoom + 1.f) * 0.5f * RTS.imageHeight;
			return RTVector3D({ sx, sy, p[2] }); // x, y screen space + z (for interpolation)
			};
		
		//line overlay:
		if (RTS.perspectiveFactor == 0 && RTS.showLines) //otherwise, requires to adapt line drawing
		{
			for (const auto& line : lines)
			{
				RTVector3D p0 = line.point1;
				RTVector3D p1 = line.point2;
				Float4 color0 = line.color1;
				Float4 color1 = line.color2;

				RTVector3D s0 = ToScreen(p0);
				RTVector3D s1 = ToScreen(p1);

				RasterizeLine(s0, s1, color0, color1, zDepths, pixelsRGBA, RTS);
			}
		}
		//std::cout << "lines=" << cntLines << "\n";
		Real tEnd = EXUstd::GetTimeInSeconds();
		if (RTS.verbose) 
		{
			GlfwRenderer::PrintDelayed("nTrigs" + EXUstd::ToString(triangles.NumberOfItems())
				+", tRayFrame=" + EXUstd::ToString(tEnd - tStartRT)
				+ ", sTree=" + EXUstd::ToString(RTS.searchTree.NumberOfCellsXYZ()));
		}
		if (rendererDebug)
		{ 
			//std::cout << "nTrigs" << triangles.NumberOfItems() << ", box3D=" << box3D << "\n";
			GlfwRenderer::PrintDelayed(" tPre=" + EXUstd::ToString(tStartRT-tStart) 
				+ "  nRT=" + EXUstd::ToString(nHits / 1e6) + "M, nRT/s=" 
				+ EXUstd::ToString(nHits / 1e6 / EXUstd::Maximum(1e-6, tEnd - tStartRT)) 
				+ "M/s");
			//without searchTree, 1 thread: 100M/S; with searchTree: 57M/s
			nHits = 0;
		}

		if (RTS.AAfactor >= 1)
		{
			static ResizableArray<unsigned char> finalImage;
			Index aaImageWidth = RTS.imageWidth / RTS.AAfactor;
			Index aaImageHeight = RTS.imageHeight / RTS.AAfactor;
			finalImage.SetNumberOfItems(aaImageWidth * aaImageHeight * RTS.RTcolorDepth);
			RTfloat sFactor = (RTfloat)(RTS.AAfactor * RTS.AAfactor)* 255.f;

			const float kernel[3][3] = {
					{ 1.f, 2.f, 1.f },
					{ 2.f, 4.f, 2.f },
					{ 1.f, 2.f, 1.f }
				};
			const float kernelSum = 16.f;
			if (RTS.AAfactor == 3) { sFactor = kernelSum * 255.f; }

			for (Index y = 0; y < aaImageHeight; ++y)
			{
				for (Index x = 0; x < aaImageWidth; ++x)
				{
					Index loIdx = (y * aaImageWidth + x) * RTS.RTcolorDepth;
					RTfloat r = 0, g = 0, b = 0;
					for (Index dy = 0; dy < (Index)RTS.AAfactor; ++dy)
					{
						for (Index dx = 0; dx < (Index)RTS.AAfactor; ++dx)
						{
							Index sx = x * (Index)RTS.AAfactor + dx;
							Index sy = y * (Index)RTS.AAfactor + dy;
							Index hiIdx = (sy * RTS.imageWidth + sx) * RTS.RTcolorDepth;
							RTfloat weight = 1.f;
							if (RTS.AAfactor == 3) { weight = kernel[dy][dx]; }
							r += weight * (int)pixelsRGBA[hiIdx + 0];
							g += weight * (int)pixelsRGBA[hiIdx + 1];
							b += weight * (int)pixelsRGBA[hiIdx + 2];
						}
					}
					
					finalImage[loIdx + 0] = FloatColorToByte(r / sFactor);
					finalImage[loIdx + 1] = FloatColorToByte(g / sFactor);
					finalImage[loIdx + 2] = FloatColorToByte(b / sFactor);
					finalImage[loIdx + 3] = 255;
				}
			}
			DrawImageRGBA(finalImage, aaImageWidth, aaImageHeight, width, height);
		}
		else
		{
			DrawImageRGBA(pixelsRGBA, RTS.imageWidth, RTS.imageHeight, width, height);
		}
	}
}

void RaytracingSettings::Initialize()
{
	//DELETE (overwritten by visualizationSettings):
	//general RTS:
	//maxReflectionDepth = 2; //2
	//maxTransparencyDepth = 2; //1

	//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	// DELETE
	//materials:
	//Exudyn's openGL default material
	//shiny material; 
	//materials[0].specular = RTVector3D({ 0.6f, 0.6f, 0.6f });
	//materials[0].shininess = 32.0f;
	//materials[0].reflectivity = 0.f;
	//materials[0].emission = RTVector3D({ 0.0f, 0.0f, 0.0f });  // Not self-luminous
	//materials[0].ior = 1.f;
	//materials[0].alpha = 1.f;

	////regular material
	//materials[1].specular = RTVector3D({ 0.3f, 0.3f, 0.3f });  // Strong white highlights
	//materials[1].shininess = 5.0f;
	//materials[1].reflectivity = 0.f;                     
	//materials[1].emission = RTVector3D({ 0.0f, 0.0f, 0.0f });  // Not self-luminous
	//materials[1].ior = 1.f;                              
	//materials[1].alpha = 1.f;  

	////metal/steel (regular, reflective):
	//materials[2].specular = RTVector3D({ 0.3f, 0.33f, 0.4f }); 
	//materials[2].shininess = 25.0f;			
	//materials[2].reflectivity = 0.1f;
	//materials[2].emission = RTVector3D({ 0.0f, 0.0f, 0.0f }); 
	//materials[2].ior = 1.f;										
	//materials[2].alpha = 1.f;

	////metal/chromium (shiny and reflective):
	//materials[3].specular = RTVector3D({ 0.6f, 0.62f, 0.67f }); 
	//materials[3].shininess = 60.0f;								
	//materials[3].reflectivity = 0.25f;
	//materials[3].emission = RTVector3D({ 0.0f, 0.0f, 0.0f }); 
	//materials[3].ior = 1.f;										
	//materials[3].alpha = 1.f;

	////transparent/glass:
	//materials[4].specular = RTVector3D({ 0.6f, 0.68f, 0.63f }); 
	//materials[4].shininess = 50.0f;								
	//materials[4].reflectivity = 0.6f;						  
	//materials[4].emission = RTVector3D({ 0.0f, 0.0f, 0.0f }); 
	//materials[4].ior = 1.5f;
	//materials[4].alpha = 0.15f;

	////transparent/plastics:
	//materials[5].specular = RTVector3D({ 0.3f, 0.3f, 0.3f });
	//materials[5].shininess = 10.0f;
	//materials[5].reflectivity = 0.2f;
	//materials[5].emission = RTVector3D({ 0.0f, 0.0f, 0.0f });
	//materials[5].ior = 1.05f;
	//materials[5].alpha = 0.3f;

	////mirror:
	//materials[6].specular = RTVector3D({ 0.2f, 0.2f, 0.2f });
	//materials[6].shininess = 50.0f;
	//materials[6].reflectivity = 0.8f;
	//materials[6].emission = RTVector3D({ 0.0f, 0.0f, 0.0f });
	//materials[6].ior = 1.f;
	//materials[6].alpha = 1.f;

	////shining/emission:
	//materials[7].specular = RTVector3D({ 0.4f, 0.4f, 0.4f });
	//materials[7].shininess = 20.0f;
	//materials[7].reflectivity = 0.3f;
	//materials[7].emission = RTVector3D({ 0.8f, 0.8f, 0.7f });
	//materials[7].ior = 1.f;
	//materials[7].alpha = 1.f;

	////user; overwritten with openGL settings:
	//materials[9].specular = RTVector3D({ 0.3f, 0.3f, 0.3f });  // Strong white highlights
	//materials[9].shininess = 5.0f;
	//materials[9].reflectivity = 0.f;
	//materials[9].emission = RTVector3D({ 0.0f, 0.0f, 0.0f });  // Not self-luminous
	//materials[9].ior = 1.f;
	//materials[9].alpha = 1.f;

	//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	//Emissive Light Source
	//specular = { 0, 0, 0 }, shininess = 0, reflectivity = 0, roughness = 1,
	//emission = { 5, 5, 5 }, ior = 1, alpha = 1
	//Simulates an area light or glowing object(like an LED or bulb)

	//Frosted Glass
	//specular = { 0.3, 0.3, 0.3 }, shininess = 5.0, reflectivity = 0.3,
	//roughness = 0.3, emission = { 0, 0, 0 }, ior = 1.5, alpha = 0.2
	//For diffusive transparent materials like sandblasted glass

	//Rubber
	//specular = { 0.05, 0.05, 0.05 }, shininess = 5.0, reflectivity = 0.0,
	//roughness = 0.6, emission = { 0, 0, 0 }, ior = 1.1, alpha = 1.0
	//Good for tires or gaskets

	//Wood(Varnished)
	//specular = { 0.15, 0.1, 0.05 }, shininess = 20.0, reflectivity = 0.05,
	//roughness = 0.2, emission = { 0, 0, 0 }, ior = 1.45, alpha = 1.0
	//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

		
	////general light:
	////backgroundColor = RTVector3D({ 0.9f,0.9f,1.f });
	//backgroundColor = RTVector3D({ 0.1f,0.1f,0.1f });
	//backgroundColorReflections = RTVector3D({ 0.5f,0.5f,0.5f }); //this is the color used for reflections and refractions
	//ambientLightColor = { 0.4f, 0.4f, 0.4f };

	////specific lights:
	//// Light 1: Above and to the right
	//lights[0].position = RTVector4D({ 2.0f, -3.0f, 5.0f, 1.0f });
	////lights[0].position = RTVector4D({ 1.f,0.5f,-1.f, 0.f });
	////lights[0].position.Normalize(); //works for directional light, which has 4th component 0

	//lights[0].ambient = 0.f; //UNUSED
	//lights[0].diffuse = 0.4f;
	//lights[0].specular = 1.f;
	//lights[0].constantAttenuation = 1.0f;
	//lights[0].linearAttenuation = 0.1f;
	//lights[0].quadraticAttenuation = 0.01f;
	//lights[0].active = true;

	//// Light 2: Behind and below
	//lights[1].position = RTVector4D({ -3.0f, -2.0f, 1.0f, 1.0f });
	//lights[1].ambient = 0.f; //UNUSED
	//lights[1].diffuse = 0.8f;
	//lights[1].specular = 0.5f;
	//lights[1].constantAttenuation = 0.5f;
	//lights[1].linearAttenuation = 0.2f;
	//lights[1].quadraticAttenuation = 0.4f;
	//lights[1].active = false;

}



//! draw a RGBA 8-bit image with given resolution
void Raytracer::DrawImageRGBA(const ResizableArray<unsigned char>& pixelsRGBA, Index imageWidth, Index imageHeight,
	Index openGLwindowWidth, Index openGLwindowHeight)
{
	const Index RTcolorDepth = 4;
	glPushAttrib(GL_ENABLE_BIT | GL_TEXTURE_BIT);
	glPushClientAttrib(GL_CLIENT_ALL_ATTRIB_BITS);

	//+++++++++++++++++++++
	//this is needed to be coherent with Render3DObjects:
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
	glStencilMask(~0); //make sure that all stencil bits are cleared
	glClearStencil(0);
	//+++++++++++++++++++++

	glDisable(GL_DEPTH_TEST);
	glDisable(GL_LIGHTING);
	glDisable(GL_BLEND);
	glEnable(GL_TEXTURE_2D);
	glColor3f(1.0f, 1.0f, 1.0f);  // Don't modulate texture color

	// --- Create (or reuse) texture and buffer
	static GLuint textureID = 0;
	static Index lastWidth = 0, lastHeight = 0;

	if (imageWidth != lastWidth || imageHeight != lastHeight ||
		textureID == 0)
	{
		//if (rendererDebug) { std::cout << "create texture" << "\n"; }
		lastWidth = imageWidth;
		lastHeight = imageHeight;

		if (textureID != 0 && glIsTexture(textureID))
		{
			glDeleteTextures(1, &textureID);
		}
		glGenTextures(1, &textureID);
	}
	glBindTexture(GL_TEXTURE_2D, textureID);

	// Fill new gradient
	unsigned char* pixelBuffer = pixelsRGBA.GetDataPointer();

	// Create new texture
	glTexImage2D(GL_TEXTURE_2D, 0, RTcolorDepth, imageWidth, imageHeight, 0,
		GL_RGBA, GL_UNSIGNED_BYTE, pixelBuffer); //only GL_RGBD with 4 colors seems to work properly
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	// --- Set up exact pixel-space orthographic projection
	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();
	glOrtho(0.0, openGLwindowWidth, 0.0, openGLwindowHeight, -1.0, 1.0);

	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();

	glViewport(0, 0, openGLwindowWidth, openGLwindowHeight);

	// --- Draw full-screen textured quad
	glBindTexture(GL_TEXTURE_2D, textureID);

	//GLint currentTex = 0;
	//glGetIntegerv(GL_TEXTURE_BINDING_2D, &currentTex);
	//std::cout << "Currently bound texture: " << currentTex << std::endl;

	glBegin(GL_QUADS);
	glTexCoord2f(0.0f, 0.0f); glVertex2i(0, 0);
	glTexCoord2f(1.0f, 0.0f); glVertex2i(openGLwindowWidth, 0);
	glTexCoord2f(1.0f, 1.0f); glVertex2i(openGLwindowWidth, openGLwindowHeight);
	glTexCoord2f(0.0f, 1.0f); glVertex2i(0, openGLwindowHeight);
	glEnd();

	// --- Restore matrices
	glMatrixMode(GL_MODELVIEW);
	glPopMatrix();
	glMatrixMode(GL_PROJECTION);
	glPopMatrix();

	glPopClientAttrib();
	glPopAttrib();
}









//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#endif //USE_GLFW_GRAPHICS

