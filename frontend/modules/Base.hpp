#pragma once
#ifndef BASE_H
#define BASE_H

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

			// Each module must have a width and height, as this specifies the size of 
			// the surface object a module will write to, and must match the dimensions
			// of the texture object the surface will read from.
			Module(int width, int height);

			// Destructor calls functions to clear CUDA objects/data
			virtual ~Module();

			// Conversion
			
			
			// Connects this module to another source module
			virtual void ConnectModule(Module* other);

			// Generates data and stores it in this object
			virtual void Generate() = 0;

			// Returns Generated data.
			virtual std::vector<float> GetData() const;

			// Gets reference to module at given index in this modules "sourceModules" container
			virtual Module& GetModule(size_t idx) const;

			// Get number of source modules connected to this object.
			virtual size_t GetSourceModuleCount() const = 0;

			// Get texture from GPU and return it as a normalized (0.0 - 1.0) vector floating point values
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
			std::pair<int, int> dims;

			// Modules that precede this module, with the back 
			// of the vector being the module immediately before 
			// this one, and the front of the vector being the initial
			// module.
			std::vector<Module*> sourceModules;
		};

		class Module3D {
			// Delete copy ctor and operator
			Module3D(const Module3D& other) = delete;
			Module3D& operator=(const Module3D& other) = delete;
			Module3D(Module3D&& other) = delete;
			Module3D& operator=(Module3D&& other) = delete;
		public:

			// Each Module3D must have a width and height, as this specifies the size of 
			// the surface object a Module3D will write to, and must match the dimensions
			// of the texture object the surface will read from.
			Module3D(int width, int height, int depth);

			// Destructor calls functions to clear CUDA objects/data
			virtual ~Module3D();

			// Conversion


			// Connects this Module3D to another source Module3D
			virtual void ConnectModule(Module3D* other);

			// Generates data and stores it in this object
			virtual void Generate() = 0;

			// Returns Generated data.
			virtual std::vector<float> GetData() const;

			std::vector<float> GetLayer(size_t idx) const;

			// Gets reference to module at given index in this modules "sourceModules" container
			virtual Module3D* GetModule(size_t idx) const;

			// Get number of source modules connected to this object.
			virtual size_t GetSourceModuleCount() const = 0;

			// Save noise at depth "idx" to output image.
			virtual void SaveToPNG(const char* name, size_t depth_idx);

			void SaveAllToPNGs(const char * base_name);

			// Tells us whether or not this module has already Generated data.
			bool Generated;

			// Each module will write self values into this, and allow other modules to read from this.
			// Allocated with managed memory.
			float* Output;

		protected:

			// Dimensions of textures.
			int3 Dimensions;

			// Modules that precede this module, with the back 
			// of the vector being the module immediately before 
			// this one, and the front of the vector being the initial
			// module.
			std::vector<Module3D*> sourceModules;
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