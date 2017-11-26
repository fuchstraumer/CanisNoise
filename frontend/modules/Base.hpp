#pragma once
#ifndef BASE_H
#define BASE_H
#include <vector>
#include <utility>
#include <memory>
/*
	
	Defines a base module class.

	Each module inherits from this, so that we can
	link modules together safely.

	This mainly involves checking for compatible parameters
	between linked modules, and chaining together generate
	commands to create the final object.

*/


namespace cnoise {

		class Module {
			// Delete copy ctor and operator
			Module(const Module& other) = delete;
			Module& operator=(const Module& other) = delete;
			Module(Module&& other) = delete;
			Module& operator=(Module&& other) = delete;
		public:


			Module(const size_t& width, const size_t& height);
			virtual ~Module();

			virtual void ConnectModule(std::shared_ptr<Module>& other);
			virtual void Generate() = 0;


			virtual std::vector<float> GetData() const;

			virtual Module& GetModule(size_t idx) const;
			virtual size_t GetSourceModuleCount() const = 0;

			virtual std::vector<float> GetDataNormalized(float upper_bound, float lower_bound) const;

			// Save current module to an image with name "name"
			virtual void SaveToPNG(const char* name);

			void SaveToPNG_16(const char * filename);

			void SaveRaw32(const char * filename);

			void SaveToTER(const char * name);

			// Tells us whether or not this module has already Generated data.
			bool Generated;

			// Each module will write self values into this, and allow other modules to read from this.
			// Allocated with managed memory.
			float* Output;

		protected:

			// Dimensions of textures.
			std::pair<size_t, size_t> dims;

			// Modules that precede this module, with the back 
			// of the vector being the module immediately before 
			// this one, and the front of the vector being the initial
			// module.
			std::vector<std::shared_ptr<Module>> sourceModules;
		};

		namespace generators {

			// Config struct for noise generators.
			struct noiseCfg {

				// Seed for the noise generator
				int Seed;
				// Frequency of the noise
				float Frequency;
				// Lacunarity controls amplitude of the noise, effectively
				float Lacunarity;
				// Controls how many octaves to use during octaved noise generation
				int Octaves;
				// Persistence controls how the amplitude of successive octaves decreases.
				float Persistence;

				noiseCfg(int s, float f, float l, int o, float p) : Seed(s), Frequency(f), Lacunarity(l), Octaves(o), Persistence(p) {}

				noiseCfg() = default;
				~noiseCfg() = default;

			};

		}
}


#endif // !BASE_H
