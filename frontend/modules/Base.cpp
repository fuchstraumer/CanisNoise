#include "Base.hpp"
#include "cuda_assert.h"
#include "../image/Image.hpp"
#include "utility/normalize.cuh"

namespace cnoise {
    
    Module::Module(const size_t& width, const size_t& height) : dims(width, height) {
            Generated = false;
            // Allocate using managed memory, so that CPU/GPU can share a single pointer.
            // Be sure to call cudaDeviceSynchronize() before accessing Output.
            cudaError_t err = cudaSuccess;
            err = cudaMallocManaged(&Output, sizeof(float) * width * height);
            cudaAssert(err);

            // Synchronize device to make sure we can access the Output pointer freely and safely.
            err = cudaDeviceSynchronize();
            cudaAssert(err);
        }

        Module::~Module() {
            cudaError_t err = cudaSuccess;
            // Synchronize device to make sure its not doing anything with the elements we wish to destroy
            err = cudaDeviceSynchronize();
            cudaAssert(err);
            // Free managed memory.
            err = cudaFree(Output);
            cudaAssert(err);
        }

        void Module::ConnectModule(const std::shared_ptr<Module>& other) {
            if (sourceModules.size() < GetSourceModuleCount()) {
                sourceModules.push_back(other);
            }
        }

        std::vector<float> Module::GetData() const{
            // Make sure to sync device before trying to get data.
            auto err = cudaDeviceSynchronize();
            cudaAssert(err);
            std::vector<float> result(Output, Output + (dims.first * dims.second));
            return result;
        }

        Module& Module::GetModule(size_t idx) const {
            // .at(idx) has bounds checking in debug modes, iirc.
            return *sourceModules.at(idx);
        }

        std::vector<float> Module::GetDataNormalized(float upper_bound, float lower_bound) const {
            cudaError_t err = cudaDeviceSynchronize();
            err = cudaDeviceSynchronize();
            cudaAssert(err);
            float* norm;
            err = cudaMallocManaged(&norm, dims.first * dims.second * sizeof(float));
            cudaAssert(err);

            NormalizeLauncher(norm, Output, dims.first, dims.second);

            err = cudaDeviceSynchronize();
            cudaAssert(err);

            auto result = std::vector<float>(norm, norm + (dims.first * dims.second));
            cudaFree(norm);
            return std::move(result);
        }

        void Module::SaveToPNG(const char * name){
            std::vector<float> rawData = GetData();
            img::ImageWriter out(dims.first, dims.second);
            out.SetRawData(rawData);
            out.WritePNG(name);
        }

        void Module::SaveToPNG_16(const char * filename) {
            std::vector<float> raw = GetData();
            img::ImageWriter out(dims.first, dims.second);
            out.SetRawData(raw);
            out.WritePNG_16(filename);
        }

        void Module::SaveRaw32(const char* filename) {
            std::vector<float> raw = GetData();
            img::ImageWriter out(dims.first, dims.second);
            out.SetRawData(raw);
            out.WriteRaw32(filename);
        }

        void Module::SaveToTER(const char * name) {
            std::vector<float> rawData = GetData();
            img::ImageWriter out(dims.first, dims.second);
            out.SetRawData(rawData);
            out.WriteTER(name);
        }

        
}

