#include "Curve.hpp"
#include "modifiers/curve.cuh"
#include <iostream>
namespace cnoise {

	namespace modifiers {

		Curve::Curve(const size_t& width, const size_t& height) : Module(width, height) {}

		Curve::Curve(const size_t& width, const size_t& height, const std::vector<ControlPoint>& init_points) : Module(width, height), controlPoints(init_points) {}

		void Curve::AddControlPoint(const float& input_val, const float& output_val){
			controlPoints.push_back(ControlPoint(input_val, output_val));
		}

		size_t Curve::GetSourceModuleCount() const {
			return 1;
		}

		const std::vector<ControlPoint>& Curve::GetControlPoints() const{
			return controlPoints;
		}

		void Curve::SetControlPoints(const std::vector<ControlPoint>& pts){
			controlPoints.clear();
			controlPoints = pts;
		}

		void Curve::ClearControlPoints(){
			controlPoints.clear();
			controlPoints.shrink_to_fit();
		}

		void Curve::Generate(){
			if (controlPoints.empty()) {
				std::cerr << "Data unprepared in a Curve module, and control point vector empty! Exiting." << std::endl;
				throw("NOISE::MODULES::MODIFIER::CURVE:L103: Control points vector empty.");
			}
			if (sourceModules.front() == nullptr) {
				std::cerr << "Did you forget to attach a module for curving to this curve module?" << std::endl;
				throw("NOISE::MODULES::MODIFIER::CURVE:L32: No source module connected.");
			}
			
			if (!sourceModules.front()->Generated) {
				sourceModules.front()->Generate();
			}
			
			CurveLauncher(Output, sourceModules.front()->Output, dims.first, dims.second, controlPoints.data(), static_cast<int>(controlPoints.size()));
			Generated = true;
		}
	}
}