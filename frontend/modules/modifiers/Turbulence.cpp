#include "Turbulence.hpp"
#include "modifiers/turbulence.cuh"
#include <iostream>
namespace cnoise {
    
    namespace modifiers {

        Turbulence::Turbulence(const size_t& width, const size_t& height, const std::shared_ptr<Module>& prev, const int& _roughness, const int& _seed, const float& _strength, const float& freq) : Module(width, height), 
            roughness(_roughness), seed(_seed), strength(_strength), frequency(freq) {
            ConnectModule(prev);
        }

        size_t Turbulence::GetSourceModuleCount() const{
            return 1;
        }

        void Turbulence::Generate(){
            if (sourceModules.front() == nullptr) {
                throw std::runtime_error("No source module for turbulence module set before attempting to generate results.");
            }
            if (!sourceModules.front()->Generated) {
                sourceModules.front()->Generate();
            }
            TurbulenceLauncher(Output, sourceModules.front()->Output, dims.first, dims.second, noiseType, roughness, seed, strength, frequency);
            Generated = true;
        }


        void Turbulence::SetStrength(const float& _strength){
            strength = _strength;
        }

        float Turbulence::GetStrength() const{
            return strength;
        }

        void Turbulence::SetSeed(int _seed){
            seed = _seed;
        }

        int Turbulence::GetSeed() const{
            return seed;
        }

        void Turbulence::SetRoughness(const int& _rough) {
            roughness = _rough;
        }

        int Turbulence::GetRoughness() const {
            return roughness;
        }

        float Turbulence::GetFrequency() const{
            return frequency;
        }

        void Turbulence::SetFrequency(const float& _freq){
            frequency = _freq;
        }

    }

}