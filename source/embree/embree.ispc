#include "../texture_channel_mask.h"
#include "disney_bsdf.ih"
#include "float3.ih"
// #include "lcg_rng.ih"
#include "lights.ih"
#include "mat4.ih"
#include "texture2d.ih"
#include "util.ih"
#include <embree3/rtcore.isph>

#include "blue_noise_rng.ih"

struct MaterialParams {
    float3 base_color;
    float metallic;

    float specular;
    float roughness;
    float specular_tint;
    float anisotropy;

    float sheen;
    float sheen_tint;
    float clearcoat;
    float clearcoat_gloss;

    float ior;
    float specular_transmission;
};

struct ISPCGeometry {
    const float4 *uniform vertex_buf;
    const uint3 *uniform index_buf;
    const float3 *uniform normal_buf;
    const float2 *uniform uv_buf;
};

struct ISPCInstance {
    const ISPCGeometry *uniform geometries;
    const float *uniform object_to_world;
    const float *uniform world_to_object;
    const uint32_t *uniform material_ids;
};

struct SceneContext {
    RTCScene scene;
    ISPCInstance *uniform instances;
    MaterialParams *uniform materials;
    QuadLight *uniform lights;
    ISPCTexture2D *uniform textures;
    uniform uint32_t num_lights;
};

struct Tile {
    uint32_t x, y;
    uint32_t width, height;
    uint32_t fb_width, fb_height;

    float *uniform color;
    float *uniform albedo;
    float *uniform shadow;
    float *uniform position;
    float *uniform depth;
    float *uniform normal;

    uint32_t *uniform curr_rng_seed;
    uint32_t *uniform prev_rng_seed;

    uint16_t *uniform ray_stats;

    float camera_x, camera_y, camera_z;

    uint32_t frame_id;

    ISPCTexture2D *uniform blue_noise;
};



float textured_scalar_param(const float x,
                            const float2 &uv,
                            const ISPCTexture2D *uniform textures)
{
    const uint32_t mask = intbits(x);
    if (IS_TEXTURED_PARAM(mask)) {
        const uint32_t tex_id = GET_TEXTURE_ID(mask);
        const uint32_t channel = GET_TEXTURE_CHANNEL(mask);
        return texture_channel(&textures[tex_id], uv, channel);
    }
    return x;
}

void unpack_material(DisneyMaterial &mat,
                     const MaterialParams *p,
                     const ISPCTexture2D *uniform textures,
                     const float2 uv)
{
    uint32_t mask = intbits(p->base_color.x);
    if (IS_TEXTURED_PARAM(mask)) {
        const uint32_t tex_id = GET_TEXTURE_ID(mask);
        mat.base_color = make_float3(texture(&textures[tex_id], uv));
    } else {
        mat.base_color = p->base_color;
    }

    mat.metallic = textured_scalar_param(p->metallic, uv, textures);
    mat.specular = textured_scalar_param(p->specular, uv, textures);
    mat.roughness = textured_scalar_param(p->roughness, uv, textures);
    mat.specular_tint = textured_scalar_param(p->specular_tint, uv, textures);
    mat.anisotropy = textured_scalar_param(p->anisotropy, uv, textures);
    mat.sheen = textured_scalar_param(p->sheen, uv, textures);
    mat.sheen_tint = textured_scalar_param(p->sheen_tint, uv, textures);
    mat.clearcoat = textured_scalar_param(p->clearcoat, uv, textures);
    mat.clearcoat_gloss = textured_scalar_param(p->clearcoat_gloss, uv, textures);
    mat.ior = textured_scalar_param(p->ior, uv, textures);
    mat.specular_transmission = textured_scalar_param(p->specular_transmission, uv, textures);
}

float3 sample_direct_light(const SceneContext *uniform scene,
                           const DisneyMaterial &mat,
                           const float3 &hit_p,
                           const float3 &n,
                           const float3 &v_x,
                           const float3 &v_y,
                           const float3 &w_o,
                           RTCIntersectContext *uniform incoherent_context,
                           QuadLight *uniform lights,
                           uniform uint32_t num_lights,
                           ISPCTexture2D *uniform blue_noise, uint32_t rng, uint32_t bounce)
{
    float3 illum = make_float3(0.f);

    uint32_t light_id = random_blue_noise(blue_noise, RNG_NEE_LIGHT_SELECTION(bounce), rng) * num_lights;
    light_id = min(light_id, num_lights - 1);
    QuadLight light = lights[light_id];

    RTCRay shadow_ray;

    // Sample the light to compute an incident light ray to this point
    {   
        float light_x = random_blue_noise(blue_noise, RNG_SUNLIGHT_X(bounce), rng);
        float light_y = random_blue_noise(blue_noise, RNG_SUNLIGHT_Y(bounce), rng);
        float3 light_pos =
            sample_quad_light_position(light, make_float2(light_x, light_y));
        float3 light_dir = light_pos - hit_p;
        float light_dist = length(light_dir);
        light_dir = normalize(light_dir);

        float light_pdf = quad_light_pdf(light, light_pos, hit_p, light_dir);
        float bsdf_pdf = disney_pdf(mat, n, w_o, light_dir, v_x, v_y);

        set_ray(shadow_ray, hit_p, light_dir, EPSILON);
        shadow_ray.tfar = light_dist;
        rtcOccludedV(scene->scene, incoherent_context, &shadow_ray);

        if (light_pdf >= EPSILON && bsdf_pdf >= EPSILON && shadow_ray.tfar > 0.f) {
            float3 bsdf = disney_brdf(mat, n, w_o, light_dir, v_x, v_y);
            float w = power_heuristic(1.f, light_pdf, 1.f, bsdf_pdf);
            illum = bsdf * light.emission * abs(dot(light_dir, n)) * w / light_pdf;
        }
    }

    // Sample the BRDF to compute a light sample as well
    {
        float3 w_i;
        float bsdf_pdf;
        float3 bsdf = sample_disney_brdf(
            mat, n, w_o, v_x, v_y, blue_noise, rng, bounce, w_i, bsdf_pdf
        );

        float light_dist;
        float3 light_pos;
        if (!all_zero(bsdf) && bsdf_pdf >= EPSILON &&
            quad_intersect(light, hit_p, w_i, light_dist, light_pos)) {
            float light_pdf = quad_light_pdf(light, light_pos, hit_p, w_i);
            if (light_pdf >= EPSILON) {
                float w = power_heuristic(1.f, bsdf_pdf, 1.f, light_pdf);
                set_ray(shadow_ray, hit_p, w_i, EPSILON);
                shadow_ray.tfar = light_dist;
                rtcOccludedV(scene->scene, incoherent_context, &shadow_ray);

                if (shadow_ray.tfar > 0.f) {
                    illum = illum + bsdf * light.emission * abs(dot(w_i, n)) * w / bsdf_pdf;
                }
            }
        }
    }

    return illum;
}


export void trace_rays(void *uniform _scene,
                       void *uniform _tile)
{
    SceneContext *uniform scene = (SceneContext * uniform) _scene;
    Tile *uniform tile = (Tile * uniform) _tile;
    uniform RTCIntersectContext context;
    rtcInitIntersectContext(&context);
    context.flags = RTC_INTERSECT_CONTEXT_FLAG_COHERENT;

    foreach (ray = 0 ... tile->width * tile->height) {
        const uint32_t px_id = ray * 3;

        tile->shadow[px_id] =  0.0;
        tile->shadow[px_id + 1] =  0.0;
        tile->shadow[px_id + 2] =  0.0;

        const uint32_t i = mod(ray, tile->width);
        const uint32_t j = ray / tile->width;

        float3 illum = make_float3(0.0);

        const float px_x = (i + tile->x) / tile->fb_width;
        const float px_y = (j + tile->y) / tile->fb_height;

        const uint32_t fb_px = ((j + tile->y) * tile->fb_width + i + tile->x) * 4;

        tile->albedo[px_id] = 0.0;
        tile->albedo[px_id + 1] = 0.0;
        tile->albedo[px_id + 2] = 0.0;

        uint32_t rng = tile->curr_rng_seed[fb_px];

        // print("rng: %\n", rng);
        // print("random: %\n", random_blue_noise(tile->blue_noise, RNG_BRDF_X(4), rng));

        // if (tile->curr_rng_seed[fb_px] == 0) {
        //     rng = get_rng((tile->x + i + (tile->y + j) * tile->fb_width),
        //                         tile->frame_id + 1);
        // } else {
        //     rng.state = tile->curr_rng_seed[fb_px];
        // }

        tile->prev_rng_seed[fb_px] = rng;

        // random_blue_noise(tile->blue_noise, make_int2(px_x, px_y), tile->curr_rng_seed[fb_px]);

        RTCRayHit path_ray;
        float3 org = make_float3(tile->position[fb_px], tile->position[fb_px + 1], tile->position[fb_px + 2]);
        {

            if (all_zero(org)) {
                continue;
            }
            float3 cam_pos = make_float3(tile->camera_x, tile->camera_y, tile->camera_z);
            float3 dir = normalize(org - cam_pos);

            set_ray_hit(path_ray, cam_pos, dir, 0.0f);
        }

        int bounce = 0;
        float3 path_throughput = make_float3(1.0);
        DisneyMaterial mat;
        // Save first even normals in case the material of the first hit has a high roughness
        float prev_roughness = 0.0;
        mat4 matrix;
        mat4 matrix_p;
        do {
            // print("{%, %, %}\n", path_ray.ray.dir_x, path_ray.ray.dir_y, path_ray.ray.dir_z);
            rtcIntersectV(scene->scene, &context, &path_ray);

            context.flags = RTC_INTERSECT_CONTEXT_FLAG_INCOHERENT;

            const int inst = path_ray.hit.instID[0];
            const int geom = path_ray.hit.geomID;
            const int prim = path_ray.hit.primID;

            const float3 w_o =
                make_float3(-path_ray.ray.dir_x, -path_ray.ray.dir_y, -path_ray.ray.dir_z);

            if (geom == RTC_INVALID_GEOMETRY_ID || inst == RTC_INVALID_GEOMETRY_ID ||
                prim == RTC_INVALID_GEOMETRY_ID) {
                if (bounce == 0) {
                    tile->albedo[px_id + 0] = 0.0;
                    tile->albedo[px_id + 1] = 0.0;
                    tile->albedo[px_id + 2] = 0.0;
                }
                break;
            }

            const float3 hit_p =
                make_float3(path_ray.ray.org_x + path_ray.ray.tfar * path_ray.ray.dir_x,
                            path_ray.ray.org_y + path_ray.ray.tfar * path_ray.ray.dir_y,
                            path_ray.ray.org_z + path_ray.ray.tfar * path_ray.ray.dir_z);

            // float3 normal = normalize(
            //     make_float3(path_ray.hit.Ng_x, path_ray.hit.Ng_y, path_ray.hit.Ng_z));

            const float2 bary = make_float2(path_ray.hit.u, path_ray.hit.v);

            const ISPCInstance *instance = &scene->instances[inst];
            const ISPCGeometry *geometry = &instance->geometries[geom];

            float2 uv = make_float2(0.f, 0.f);
            const uint3 indices = geometry->index_buf[prim];

            float3 na = geometry->normal_buf[indices.x];
            float3 nb = geometry->normal_buf[indices.y];
            float3 nc = geometry->normal_buf[indices.z];

            float3 normal = (1.f - bary.x - bary.y) * na + bary.x * nb + bary.y * nc;

            if (geometry->uv_buf) {
                float2 uva = geometry->uv_buf[indices.x];
                float2 uvb = geometry->uv_buf[indices.y];
                float2 uvc = geometry->uv_buf[indices.z];
                uv = (1.f - bary.x - bary.y) * uva + bary.x * uvb + bary.y * uvc;
            }

            // Transform the normal back to world space
            load_mat4(matrix, instance->world_to_object);
            transpose(matrix);
            normal = normalize(mul(matrix, normal));

            unpack_material(
                mat, &scene->materials[instance->material_ids[geom]], scene->textures, uv);
            
            // First even normals
            if (bounce == 0) {
                if (mat.roughness < 0.4) {
                    RTCRayHit normal_ray;
                    float3 normal_ray_dir = (-1 * w_o) - 2 * normal * (-1 * w_o * normal);
                    set_ray_hit(normal_ray, hit_p, normal_ray_dir, EPSILON);
                    rtcIntersectV(scene->scene, &context, &normal_ray);

                    const int inst_p = normal_ray.hit.instID[0];
                    const int geom_p = normal_ray.hit.geomID;
                    const int prim_p = normal_ray.hit.primID;

                    if (!(geom_p == RTC_INVALID_GEOMETRY_ID || inst_p == RTC_INVALID_GEOMETRY_ID ||
                        prim_p == RTC_INVALID_GEOMETRY_ID)) {
                        const ISPCInstance *instance_p = &scene->instances[inst_p];
                        const ISPCGeometry *geometry_p = &instance_p->geometries[geom_p];

                        float3 na_p = geometry_p->normal_buf[geometry_p->index_buf[prim_p].x];
                        float3 nb_p = geometry_p->normal_buf[geometry_p->index_buf[prim_p].y];
                        float3 nc_p = geometry_p->normal_buf[geometry_p->index_buf[prim_p].z];

                        const float2 bary_p = make_float2(normal_ray.hit.u, normal_ray.hit.v);

                        float3 normal_p = (1.f - bary_p.x - bary_p.y) * na_p + bary_p.x * nb_p + bary_p.y * nc_p;

                        load_mat4(matrix_p, instance_p->world_to_object);
                        transpose(matrix_p);
                        normal_p = normalize(mul(matrix_p, normal_p));

                        // tile->normal[fb_px + 0] = normal_p.x;
                        // tile->normal[fb_px + 1] = normal_p.y;
                        // tile->normal[fb_px + 2] = normal_p.z;

                        // tile->depth[fb_px / 4] = 0.0f;
                    }
                }
                

                tile->albedo[px_id + 0] = mat.base_color.x;
                tile->albedo[px_id + 1] = mat.base_color.y;
                tile->albedo[px_id + 2] = mat.base_color.z;
            }

            // Direct light sampling
            float3 v_x, v_y;
            if (mat.specular_transmission == 0.f && dot(w_o, normal) < 0.0) {
                normal = neg(normal);
            }

            ortho_basis(v_x, v_y, normal);
            illum = illum + path_throughput * sample_direct_light(scene,
                                                                    mat,
                                                                    hit_p,
                                                                    normal,
                                                                    v_x,
                                                                    v_y,
                                                                    w_o,
                                                                    &context,
                                                                    scene->lights,
                                                                    scene->num_lights,
                                                                    tile->blue_noise,
                                                                    rng,
                                                                    bounce);

            // Sample the BSDF to continue the ray
            float pdf;
            float3 w_i;
            float3 bsdf = sample_disney_brdf(mat, normal, w_o, v_x, v_y, tile->blue_noise, rng, bounce, w_i, pdf);
            if (pdf == 0.f || all_zero(bsdf)) {
                break;
            }
            path_throughput = path_throughput * bsdf * abs(dot(w_i, normal)) / pdf;

            // Trace the ray continuing the path
            set_ray_hit(path_ray, hit_p, w_i, EPSILON);
            bounce++;

            // Russian roulette termination
            if (bounce > 3) {
                const float q = max(0.05f,
                                    1.f - max(path_throughput.x,
                                                max(path_throughput.y, path_throughput.z)));
                float termination = random_blue_noise(tile->blue_noise, RNG_TERMINATION(bounce), rng);
                if (termination < q) {
                    break;
                }
                path_throughput = path_throughput / (1.f - q);
            }
        } while (bounce < 3);

        tile->shadow[px_id + 0] = (tile->albedo[px_id + 0] < 0.03) ? 0.0 : (illum.x / tile->albedo[px_id + 0]);
        tile->shadow[px_id + 1] = (tile->albedo[px_id + 1] < 0.03) ? 0.0 : (illum.y / tile->albedo[px_id + 1]);
        tile->shadow[px_id + 2] = (tile->albedo[px_id + 2] < 0.03) ? 0.0 : (illum.z / tile->albedo[px_id + 2]);
        
        // tile->shadow[px_id + 0] = (tile->shadow[px_id + 0] > 1.4) ? 1.4 : tile->shadow[px_id + 0];
        // tile->shadow[px_id + 1] = (tile->shadow[px_id + 1] > 1.4) ? 1.4 : tile->shadow[px_id + 1];
        // tile->shadow[px_id + 2] = (tile->shadow[px_id + 2] > 1.4) ? 1.4 : tile->shadow[px_id + 2];
        // tile->shadow[px_id + 0] = tile->position[fb_px + 0];
        // tile->shadow[px_id + 1] = tile->position[fb_px + 1];
        // tile->shadow[px_id + 2] = tile->position[fb_px + 2];
    }
}

// Convert the RGBF32 tile to sRGB and write it to the RGBA8 framebuffer
export void tile_to_buffer(void *uniform _tile, uniform float *uniform shadow, uniform float *uniform albedo)
{
    Tile *uniform tile = (Tile * uniform) _tile;
    foreach (i = 0 ... tile->width, j = 0 ... tile->height) {
        const uint32_t tile_px = (j * tile->width + i) * 3;
        const uint32_t fb_px = ((j + tile->y) * tile->fb_width + i + tile->x) * 4;

        shadow[fb_px] = tile->shadow[tile_px];
        shadow[fb_px + 1] = tile->shadow[tile_px + 1];
        shadow[fb_px + 2] = tile->shadow[tile_px + 2];
        shadow[fb_px + 3] = 1.0f;

        albedo[fb_px] = tile->albedo[tile_px];
        albedo[fb_px + 1] = tile->albedo[tile_px + 1];
        albedo[fb_px + 2] = tile->albedo[tile_px + 2];
        albedo[fb_px + 3] = 1.0f;
    }
}
