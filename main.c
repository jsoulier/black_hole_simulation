#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>

#define THREADS 16
#define PAN 0.002f
#define ZOOM 25.0e9f
#define FOV (60.0f * SDL_PI_F / 180.0f)
#define CLAMP (SDL_PI_F / 2.0f - 0.01f)
#define LIGHT 299792458.0f
#define GRAVITY 6.67430e-11f
#define MASS 8.54e36f
#define RADIUS (2.0f * GRAVITY * MASS / (LIGHT * LIGHT))
#define OBJECTS 3

typedef struct Config
{
    float pitch;
    float yaw;
    float camera_distance;
    float tan_half_fov;
    Uint32 object_count;
    float disk_r1;
    float disk_r2;
} Config;

typedef struct Object
{
    float x;
    float y;
    float z;
    float radius;
    float r;
    float g;
    float b;
    float mass;
} Object;

static SDL_GPUComputePipeline* CreateComputePipeline(SDL_GPUDevice* device)
{
    SDL_GPUShaderFormat formats = SDL_GetGPUShaderFormats(device);
    const char* path;
    const char* entrypoint;
    SDL_GPUShaderFormat format;
    if (formats & SDL_GPU_SHADERFORMAT_SPIRV)
    {
        path = "shader.comp.spv";
        entrypoint = "main";
        format = SDL_GPU_SHADERFORMAT_SPIRV;
    }
    else if (formats & SDL_GPU_SHADERFORMAT_DXIL)
    {
        path = "shader.comp.dxil";
        entrypoint = "main";
        format = SDL_GPU_SHADERFORMAT_DXIL;
    }
    else if (formats & SDL_GPU_SHADERFORMAT_MSL)
    {
        path = "shader.comp.msl";
        entrypoint = "main0";
        format = SDL_GPU_SHADERFORMAT_MSL;
    }
    else
    {
        SDL_Log("No supported shader format");
        return NULL;
    }
    size_t size;
    void* data = SDL_LoadFile(path, &size);
    if (!data)
    {
        SDL_Log("Failed to load shader: %s", SDL_GetError());
        return NULL;
    }
    SDL_GPUComputePipelineCreateInfo cpci = {0};
    cpci.code = data;
    cpci.code_size = size;
    cpci.entrypoint = entrypoint;
    cpci.format = format;
    cpci.num_readonly_storage_buffers = 1;
    cpci.num_readwrite_storage_textures = 1;
    cpci.num_uniform_buffers = 1;
    cpci.threadcount_x = THREADS;
    cpci.threadcount_y = THREADS;
    cpci.threadcount_z = 1;
    SDL_GPUComputePipeline* pipeline = SDL_CreateGPUComputePipeline(device, &cpci);
    SDL_free(data);
    if (!pipeline)
    {
        SDL_Log("Failed to create compute pipeline: %s", SDL_GetError());
        return NULL;
    }
    return pipeline;
}

static SDL_GPUBuffer* CreateObjects(SDL_GPUDevice* device)
{
    Object objects[OBJECTS] =
    {{
        .x = 4.0e11f,
        .y = 0.0f,
        .z = 0.0f,
        .radius = 4.0e10f,
        .r = 1.0f,
        .g = 1.0f,
        .b = 0.0f,
        .mass = 1.98892e30f,
    },
    {
        .x = 0.0f,
        .y = 0.0f,
        .z = 4.0e11f,
        .radius = 4.0e10f,
        .r = 1.0f,
        .g = 0.0f,
        .b = 0.0f,
        .mass = 1.98892e30f,
    },
    {
        .x = 0.0f,
        .y = 0.0f,
        .z = 0.0f,
        .radius = RADIUS,
        .r = 0.0f,
        .g = 0.0f,
        .b = 0.0f,
        .mass = MASS,
    }};
    SDL_GPUTransferBufferCreateInfo tbci = {0};
    tbci.usage = SDL_GPU_TRANSFERBUFFERUSAGE_UPLOAD;
    tbci.size = sizeof(objects);
    SDL_GPUBufferCreateInfo bci = {0};
    bci.usage = SDL_GPU_BUFFERUSAGE_COMPUTE_STORAGE_READ;
    bci.size = sizeof(objects);
    SDL_GPUTransferBuffer* tbo = SDL_CreateGPUTransferBuffer(device, &tbci);
    SDL_GPUBuffer* sbo = SDL_CreateGPUBuffer(device, &bci);
    if (!tbo || !sbo)
    {
        SDL_Log("Failed to create buffer(s): %s", SDL_GetError());
        return NULL;
    }
    void* data = SDL_MapGPUTransferBuffer(device, tbo, false);
    if (!data)
    {
        SDL_Log("Failed to map transfer buffer: %s", SDL_GetError());
        return NULL;
    }
    SDL_memcpy(data, objects, sizeof(objects));
    SDL_UnmapGPUTransferBuffer(device, tbo);
    SDL_GPUCommandBuffer* command_buffer = SDL_AcquireGPUCommandBuffer(device);
    if (!command_buffer)
    {
        SDL_Log("Failed to acquire command buffer: %s", SDL_GetError());
        return NULL;
    }
    SDL_GPUCopyPass* copy_pass = SDL_BeginGPUCopyPass(command_buffer);
    if (!copy_pass)
    {
        SDL_Log("Failed to begin copy pass: %s", SDL_GetError());
        return NULL;
    }
    SDL_GPUTransferBufferLocation src = {0};
    SDL_GPUBufferRegion dst = {0};
    src.transfer_buffer = tbo;
    dst.buffer = sbo;
    dst.size = sizeof(objects);
    SDL_UploadToGPUBuffer(copy_pass, &src, &dst, false);
    SDL_EndGPUCopyPass(copy_pass);
    SDL_SubmitGPUCommandBuffer(command_buffer);
    SDL_ReleaseGPUTransferBuffer(device, tbo);
    return sbo;
}

int main(int argc, char** argv)
{
    Config config =
    {
        .camera_distance = 1.0e11f,
        .tan_half_fov = SDL_tanf(FOV * 0.5f),
        .object_count = OBJECTS,
        .disk_r1 = RADIUS * 2.2f,
        .disk_r2 = RADIUS * 5.2f,
    };
#ifndef NDEBUG
    SDL_SetLogPriorities(SDL_LOG_PRIORITY_VERBOSE);
#endif
    SDL_SetAppMetadata("Black Hole Simulation", NULL, NULL);
    if (!SDL_Init(SDL_INIT_VIDEO))
    {
        SDL_Log("Failed to initialize SDL: %s", SDL_GetError());
        return 1;
    }
    SDL_Window* window = SDL_CreateWindow("Black Hole Simulation", 960, 720, SDL_WINDOW_RESIZABLE);
    if (!window)
    {
        SDL_Log("Failed to create window: %s", SDL_GetError());
        return 1;
    }
#ifndef NDEBUG
    SDL_GPUDevice* device = SDL_CreateGPUDevice(SDL_GPU_SHADERFORMAT_DXIL | SDL_GPU_SHADERFORMAT_SPIRV | SDL_GPU_SHADERFORMAT_MSL, true, NULL);
#else
    SDL_GPUDevice* device = SDL_CreateGPUDevice(SDL_GPU_SHADERFORMAT_DXIL | SDL_GPU_SHADERFORMAT_SPIRV | SDL_GPU_SHADERFORMAT_MSL, false, NULL);
#endif
    if (!device)
    {
        SDL_Log("Failed to create device: %s", SDL_GetError());
        return 1;
    }
    if (!SDL_ClaimWindowForGPUDevice(device, window))
    {
        SDL_Log("Failed to create swapchain: %s", SDL_GetError());
        return 1;
    }
    SDL_GPUComputePipeline* pipeline = CreateComputePipeline(device);
    if (!pipeline)
    {
        SDL_Log("Failed to create compute pipeline");
        return 1;
    }
    SDL_GPUTexture* texture = NULL;
    Uint32 texture_width = 0;
    Uint32 texture_height = 0;
    SDL_GPUBuffer* objects = CreateObjects(device);
    if (!objects)
    {
        SDL_Log("Failed to create objects");
        return 1;
    }
    bool running = true;
    while (running)
    {
        SDL_Event event;
        while (SDL_PollEvent(&event))
        {
            switch (event.type)
            {
            case SDL_EVENT_MOUSE_WHEEL:
                config.camera_distance = SDL_max(1.0f, config.camera_distance - event.wheel.y * ZOOM);
                break;
            case SDL_EVENT_MOUSE_MOTION:
                if (event.motion.state & SDL_BUTTON_LMASK)
                {
                    config.yaw += event.motion.xrel * PAN;
                    config.pitch = SDL_clamp(config.pitch + event.motion.yrel * PAN, -CLAMP, CLAMP);
                }
                break;
            case SDL_EVENT_QUIT:
                running = false;
                break;
            }
        }
        SDL_GPUCommandBuffer* command_buffer = SDL_AcquireGPUCommandBuffer(device);
        if (!command_buffer)
        {
            SDL_Log("Failed to acquire command buffer: %s", SDL_GetError());
            continue;
        }
        SDL_GPUTexture* swapchain;
        Uint32 width;
        Uint32 height;
        if (!SDL_WaitAndAcquireGPUSwapchainTexture(command_buffer, window, &swapchain, &width, &height))
        {
            SDL_Log("Failed to acquire swapchain texture: %s", SDL_GetError());
            SDL_CancelGPUCommandBuffer(command_buffer);
            continue;
        }
        if (!swapchain || !width || !height)
        {
            SDL_SubmitGPUCommandBuffer(command_buffer);
            continue;
        }
        if (texture_width != width || texture_height != height)
        {
            SDL_ReleaseGPUTexture(device, texture);
            SDL_GPUTextureCreateInfo tci = {0};
            tci.usage = SDL_GPU_TEXTUREUSAGE_COMPUTE_STORAGE_WRITE | SDL_GPU_TEXTUREUSAGE_SAMPLER;
            tci.type = SDL_GPU_TEXTURETYPE_2D;
            tci.format = SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UNORM;
            tci.width = width;
            tci.height = height;
            tci.num_levels = 1;
            tci.layer_count_or_depth = 1;
            texture = SDL_CreateGPUTexture(device, &tci);
            if (!texture)
            {
                SDL_Log("Failed to create texture: %s", SDL_GetError());
                SDL_SubmitGPUCommandBuffer(command_buffer);
                running = false;
                continue;
            }
            texture_width = width;
            texture_height = height;
        }
        SDL_PushGPUComputeUniformData(command_buffer, 0, &config, sizeof(config));
        SDL_GPUStorageTextureReadWriteBinding stb = {0};
        stb.texture = texture;
        SDL_GPUComputePass* compute_pass = SDL_BeginGPUComputePass(command_buffer, &stb, 1, NULL, 0);
        if (!compute_pass)
        {
            SDL_Log("Failed to begin compute pass: %s", SDL_GetError());
            SDL_SubmitGPUCommandBuffer(command_buffer);
            continue;
        }
        SDL_BindGPUComputePipeline(compute_pass, pipeline);
        SDL_BindGPUComputeStorageBuffers(compute_pass, 0, &objects, 1);
        SDL_DispatchGPUCompute(compute_pass, (width + THREADS - 1) / THREADS, (height + THREADS - 1) / THREADS, 1);
        SDL_EndGPUComputePass(compute_pass);
        SDL_GPUBlitInfo blit = {0};
        blit.source.texture = texture;
        blit.source.w = width;
        blit.source.h = height;
        blit.destination.texture = swapchain;
        blit.destination.w = width;
        blit.destination.h = height;
        SDL_BlitGPUTexture(command_buffer, &blit);
        SDL_SubmitGPUCommandBuffer(command_buffer);
    }
    SDL_HideWindow(window);
    SDL_ReleaseGPUBuffer(device, objects);
    SDL_ReleaseGPUTexture(device, texture);
    SDL_ReleaseGPUComputePipeline(device, pipeline);
    SDL_ReleaseWindowFromGPUDevice(device, window);
    SDL_DestroyGPUDevice(device);
    SDL_DestroyWindow(window);
    SDL_Quit();
    return 0;
}
