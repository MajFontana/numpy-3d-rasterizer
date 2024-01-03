import pygame
import numpy
import time
import math
import os



WSIZE = (700, 700)
FPS = 60
FLOAT_TYPE = numpy.float32
INT_TYPE = numpy.int32



def to_4d(points):
    points = numpy.concatenate([points, numpy.ones([points.shape[0], 1], dtype=FLOAT_TYPE)], axis=1)
    return points

def perspective_divide(points):
    return points / points[:, 3:4]



class Transform:

    def __init__(self):
        self.matrix = numpy.eye(4, dtype=FLOAT_TYPE)

    def rotate_intrinsic(self, rotation):
        rotation = numpy.array(rotation)
        cos_rotation = numpy.cos(rotation)
        sin_rotation = numpy.sin(rotation)
        
        yaw_matrix = numpy.array([[cos_rotation[0], 0, -sin_rotation[0], 0],
                                  [0, 1, 0, 0],
                                  [sin_rotation[0], 0, cos_rotation[0], 0],
                                  [0, 0, 0, 1]], dtype=FLOAT_TYPE)
        pitch_matrix = numpy.array([[1, 0, 0, 0],
                                    [0, cos_rotation[1], -sin_rotation[1], 0],
                                    [0, sin_rotation[1], cos_rotation[1], 0],
                                    [0, 0, 0, 1]], dtype=FLOAT_TYPE)
        roll_matrix = numpy.array([[cos_rotation[2], -sin_rotation[2], 0, 0],
                                   [sin_rotation[2], cos_rotation[2], 0, 0],
                                   [0, 0, 1, 0],
                                   [0, 0, 0, 1]], dtype=FLOAT_TYPE)

        self.matrix = self.matrix @ yaw_matrix @ pitch_matrix @ roll_matrix
        return self

    def rotate_intrinsic_reverse(self, rotation):
        rotation = numpy.array(rotation)
        cos_rotation = numpy.cos(rotation)
        sin_rotation = numpy.sin(rotation)
        
        yaw_matrix = numpy.array([[cos_rotation[0], 0, -sin_rotation[0], 0],
                                  [0, 1, 0, 0],
                                  [sin_rotation[0], 0, cos_rotation[0], 0],
                                  [0, 0, 0, 1]], dtype=FLOAT_TYPE)
        pitch_matrix = numpy.array([[1, 0, 0, 0],
                                    [0, cos_rotation[1], -sin_rotation[1], 0],
                                    [0, sin_rotation[1], cos_rotation[1], 0],
                                    [0, 0, 0, 1]], dtype=FLOAT_TYPE)
        roll_matrix = numpy.array([[cos_rotation[2], -sin_rotation[2], 0, 0],
                                   [sin_rotation[2], cos_rotation[2], 0, 0],
                                   [0, 0, 1, 0],
                                   [0, 0, 0, 1]], dtype=FLOAT_TYPE)

        self.matrix = self.matrix @ roll_matrix @ pitch_matrix @ yaw_matrix
        return self

    def rotate_extrinsic(self, rotation):
        rotation = numpy.array(rotation)
        cos_rotation = numpy.cos(rotation)
        sin_rotation = numpy.sin(rotation)
        
        yaw_matrix = numpy.array([[cos_rotation[0], 0, -sin_rotation[0], 0],
                                  [0, 1, 0, 0],
                                  [sin_rotation[0], 0, cos_rotation[0], 0],
                                  [0, 0, 0, 1]], dtype=FLOAT_TYPE)
        pitch_matrix = numpy.array([[1, 0, 0, 0],
                                    [0, cos_rotation[1], -sin_rotation[1], 0],
                                    [0, sin_rotation[1], cos_rotation[1], 0],
                                    [0, 0, 0, 1]], dtype=FLOAT_TYPE)
        roll_matrix = numpy.array([[cos_rotation[2], -sin_rotation[2], 0, 0],
                                   [sin_rotation[2], cos_rotation[2], 0, 0],
                                   [0, 0, 1, 0],
                                   [0, 0, 0, 1]], dtype=FLOAT_TYPE)

        self.matrix = self.matrix @ yaw_matrix @ pitch_matrix @ roll_matrix
        return self

    def translate(self, translation):
        translation_matrix = numpy.array([[1, 0, 0, 0],
                                          [0, 1, 0, 0],
                                          [0, 0, 1, 0],
                                          [*translation, 1]], dtype=FLOAT_TYPE)

        self.matrix = self.matrix @ translation_matrix
        return self

    def scale(self, scaling):
        scaling_matrix = numpy.array([[scaling[0], 0, 0, 0],
                                      [0, scaling[1], 0, 0],
                                      [0, 0, scaling[2], 0],
                                      [0, 0, 0, 1]], dtype=FLOAT_TYPE)

        self.matrix = self.matrix @ scaling_matrix
        return self

    def project(self, field_of_view, z_range):
        z_scale = z_range[1] - z_range[0]
        
        projection_matrix = numpy.array([[1 / numpy.tan(field_of_view[0] / 2), 0, 0, 0],
                                         [0, 1 / numpy.tan(field_of_view[1] / 2), 0, 0],
                                         [0, 0, z_range[1] / z_scale, 1],
                                         [0, 0, -z_range[1] * z_range[0] / z_scale, 0]], dtype=FLOAT_TYPE)

        self.matrix = self.matrix @ projection_matrix
        return self

    def apply(self, points):
        return points @ self.matrix



class Object:

    def __init__(self, ):
        self.vertices = numpy.zeros([0, 4], dtype=FLOAT_TYPE)
        
        self.edges = numpy.zeros([0, 2], dtype=INT_TYPE)
        #self.triangles = numpy.zeros([0, 3], dtype=INT_TYPE)

        self.position = numpy.zeros(3, dtype=FLOAT_TYPE)
        self.rotation = numpy.zeros(3, dtype=FLOAT_TYPE)
        self.scale = numpy.ones(3, dtype=FLOAT_TYPE)



class Cube(Object):

    def __init__(self):
        super().__init__()
        
        self.vertices = to_4d(numpy.array([[-1, -1, -1],
                                   [-1, -1, 1],
                                   [-1, 1, -1],
                                   [-1, 1, 1],
                                   [1, -1, -1],
                                   [1, -1, 1],
                                   [1, 1, -1],
                                   [1, 1, 1]], dtype=self.vertices.dtype) / 2)

        self.edges = numpy.array([[0, 1],
                                  [1, 5],
                                  [5, 4],
                                  [4, 0],
                                  [2, 3],
                                  [3, 7],
                                  [7, 6],
                                  [6, 2],
                                  [0, 2],
                                  [1, 3],
                                  [5, 7],
                                  [4, 6]], dtype=self.edges.dtype)



class Grid(Object):

    def __init__(self, divisions):
        super().__init__()
        
        vertex_list = []
        edge_list = []
        for i in range(divisions):
            vertex_list += [[i, 0, 0], [i, 0, divisions - 1], [0, 0, i], [divisions - 1, 0, i]]
            edge_list += [[i * 4 , i * 4 + 1], [i * 4 + 2, i * 4 + 3]]
            
        vertices = to_4d(numpy.array(vertex_list, dtype=self.vertices.dtype))
        transform = Transform().scale([1 / (i - 1), 1, 1 / (i - 1)]).translate([-0.5, 0, -0.5])
        self.vertices = transform.apply(vertices)
        
        self.edges = numpy.array(edge_list, dtype=self.edges.dtype)



def lonlat_to_3d(coords, radius):
    lon = numpy.radians(coords[:, 0:1])
    lat = numpy.radians(coords[:, 1:2])
    x = numpy.cos(lat) * numpy.cos(lon)
    z = numpy.cos(lat) * numpy.sin(lon)
    y = numpy.sin(lat)
    pt = numpy.concatenate([x, y, z], axis=1) * radius
    return pt



# Constructs a wireframe model of Earth from a country border dataset
# https://thematicmapping.org/downloads/world_borders.php
class Earth(Object):

    def __init__(self):
        super().__init__()

        points = []
        edges = []
        off = 0

        import shapefile
        sf = shapefile.Reader(r"C:\Users\user\Downloads\TM_WORLD_BORDERS_SIMPL-0.3.zip", encoding="latin-1")
        for country in sf.shapes():
            parts = list(country.parts) + [len(country.points)]
            size = 0
            for i in range(len(parts) - 1):
                start = parts[i]
                end = parts[i + 1]
                if end > start:
                    lonlat = numpy.array(country.points[start:end], dtype=self.vertices.dtype)
                    points.append(lonlat)
                    lines = [[j, j + 1] for j in range(off + start, off + end)]
                    lines[-1][1] = off + start
                    edges.append(numpy.array(lines, dtype=self.edges.dtype))
                    size += len(lonlat)
            off += size

        self.vertices = to_4d(lonlat_to_3d(numpy.concatenate(points), 10))
        self.edges = numpy.concatenate(edges)

        #loc = [30.0, -100.0]
        #rot = numpy.radians(numpy.array([loc[1], loc[0], 0]))
        #rot[0] += numpy.pi / 2
        #rot[1] -= numpy.pi / 2
        #self.vertices = Transform().rotate_intrinsic(rot).apply(self.vertices)



class Line(Object):

    def __init__(self):
        super().__init__()
        
        self.vertices = to_4d(numpy.array([[-0.5, 0, 0],
                                           [0.5, 0, 0]], dtype=self.vertices.dtype))
        
        self.edges = numpy.array([[0, 1]], dtype=self.edges.dtype)



class Camera:

    def __init__(self):
        self.position = numpy.zeros(3, dtype=FLOAT_TYPE)
        self.rotation = numpy.zeros(3, dtype=FLOAT_TYPE)
        self.field_of_view = numpy.array([numpy.pi / 2, numpy.pi / 2], dtype=FLOAT_TYPE)
        self.z_range = numpy.array([0.1, 100], dtype=FLOAT_TYPE)

        sqr2 = math.sqrt(2)
        self.clip_planes = []
        self.clip_planes.append([sqr2, 0, 0, sqr2])
        self.clip_planes.append([-sqr2, 0, 0, sqr2])
        self.clip_planes.append([0, sqr2, 0, sqr2])
        self.clip_planes.append([0, -sqr2, 0, sqr2])
        self.clip_planes.append([0, 0, sqr2, sqr2])
        self.clip_planes.append([0, 0, -sqr2, sqr2])
        self.clip_planes.append([0, 0, 0, 1])
        self.clip_planes = numpy.array(self.clip_planes, dtype=FLOAT_TYPE)

    def render(self, objects, surface):
        t1 = time.time()
        ssize = surface.get_size()

        t6 = time.time()
        screen.fill((0, 0, 0))
        t7 = time.time()

        offsetvector = numpy.array([1, -1, 0], dtype=FLOAT_TYPE)
        scalevector = numpy.array([ssize[0] / 2 - 1, -(ssize[1] / 2 - 1), 1, 1], dtype=FLOAT_TYPE)
        screenspace_transform = Transform().translate(offsetvector).scale(scalevector)
        t8 = time.time()

        for obj in objects:
            worldspace_transform = Transform().scale(obj.scale).rotate_extrinsic(obj.rotation).translate(obj.position)
            clipspace_transform = worldspace_transform.translate(-self.position).rotate_intrinsic(-self.rotation).project(self.field_of_view, self.z_range)
            clipspace = clipspace_transform.apply(obj.vertices)
            t2 = time.time()

            clipspace_edges = numpy.take(clipspace, obj.edges.reshape([obj.edges.shape[0] * 2]), axis=0).reshape([*obj.edges.shape, 4])
            t3 = time.time()

            A = numpy.dot(clipspace_edges[:, 0, :], self.clip_planes.T)
            B = numpy.dot(clipspace_edges[:, 1, :], self.clip_planes.T)
            a = A > 0
            b = B > 0
            passmask = numpy.all(a, axis=1) & numpy.all(b, axis=1)
            rejectmask = numpy.any(numpy.logical_not(a | b), axis=1)
            passed_edges = clipspace_edges[passmask, :, :]
            clipspace_edges = clipspace_edges[numpy.logical_not(passmask | rejectmask), :, :]
            t11 = time.time()

            deltas = clipspace_edges[:, 1, :] - clipspace_edges[:, 0, :]
            dots = numpy.dot(deltas, self.clip_planes.T)
            inmask = dots > 0
            outmask = dots < 0
            ds = numpy.dot(-clipspace_edges[:, 0, :], self.clip_planes.T) / dots
            inints = numpy.nan_to_num(ds * inmask)
            outints = numpy.nan_to_num(ds * outmask, nan=1) + inmask
            startds = numpy.max(inints, axis=1)
            endds = numpy.min(outints, axis=1)
            clipds = numpy.concatenate([startds[:, None, None], endds[:, None, None]], axis=1)
            valid = endds > startds
            clip = (clipspace_edges[:, 0:1, :] + clipds * deltas[:, None, :])[valid, :, :]
            t9 = time.time()
            
            clip = numpy.concatenate([clip, passed_edges], axis=0)
            flat = clip.reshape([clip.shape[0] * 2, 4])
            per = perspective_divide(flat)
            t10 = time.time()

            coords = screenspace_transform.apply(per)
            pos = numpy.round(coords[: :, :2]).astype(INT_TYPE)
            unflat = pos.reshape([pos.shape[0] // 2, 2, 2])
            t4 = time.time()

            pixels = pygame.surfarray.pixels3d(surface)
            lengthx = unflat[:, 1, 0] - unflat[:, 0, 0]
            lengthy = unflat[:, 1, 1] - unflat[:, 0, 1]

            pointmask = (lengthx == 0) & (lengthy == 0)
            points = unflat[pointmask, 0, :]
            pixels[points[:, 0], points[:, 1]] = 255

            nopoint = ~pointmask
            alengthx = numpy.absolute(lengthx)
            alengthy = numpy.absolute(lengthy)
            xmask = alengthx >= alengthy
            ymask = ~xmask
            xmask &= nopoint
            ymask &= nopoint

            for lvstart, lvend in [[1, 2], [2, 5], [5, 11], [11, 101], [101, ssize[0]]]:
                for dim in range(2):
                    dimb = -dim + 1
                    ala = [alengthx, alengthy][dim]
                    mask = [xmask, ymask][dim] & (lvstart <= ala) & (ala < lvend)
                    lines = unflat[mask, :, :]
                    lx = lengthx[mask]
                    ly = lengthy[mask]
                    la = [lx, ly][dim]
                    lb = [ly, lx][dim]
                    ala = ala[mask]
                    
                    k = lb.astype(FLOAT_TYPE) / la.astype(FLOAT_TYPE)
                    field = numpy.empty([lines.shape[0], lvend, 2], dtype=INT_TYPE)
                    field[:, :, dim] = numpy.arange(lvend, dtype=INT_TYPE)[None, :]
                    hmask = field[:, :, dim] >= ala[:, None]
                    field[la < 0, :, dim] *= -1
                    field[:, :, dimb] = field[:, :, dim] * k[:, None] + lines[:, 0, dimb:dimb + 1]
                    field[:, :, dim] += lines[:, 0, dim:dim + 1]
                    field[:, :, :] = numpy.maximum(numpy.minimum(field[:, :, :], ssize[dim] - 1), 0)
                    field[hmask, :] = 0
                    field = field.reshape([field.shape[0] * field.shape[1], 2])
                    pixels[field[:, 0], field[:, 1]] = 255
            t5 = time.time()

        """
        global lt
        if time.time() - lt > 0.1:
            lt = time.time()
            total = t5 - t1
            d1 = t7 - t6
            d2 = t8 - t7
            d3 = t2 - t8
            d4 = t3 - t2
            d7 = t9 - t11
            d8 = t10 - t9
            d5 = t4 - t9
            d10 = t5 - t4
            d9 = t11 - t3
            totallen = len(obj.edges)
            passlen = len(passed_edges)
            vislen = len(unflat)
            rejlen = totallen - vislen
            cliplen = vislen - passlen

            os.system("cls")
            print(f"{str(totallen) + ' edges': <24} ({str(vislen) + ' visible': <16}{str(rejlen) + ' invisible': <16}{str(cliplen) + ' clipped': <16})")
            print()
            print("TOTAL: %.2f ms (%i FPS)" % (total * 1000, 1 // total))
            print("Screen fill:                %03i ms (%04.1f %%)" % (int(d1 * 1000),  100 * d1 / total))
            print("Screenspace transform init: %03i ms (%04.1f %%)" % (int(d2 * 1000),  100 * d2 / total))
            print("Clipspace transform:        %03i ms (%04.1f %%)" % (int(d3 * 1000),  100 * d3 / total))
            print("Vertex take:                %03i ms (%04.1f %%)" % (int(d4 * 1000),  100 * d4 / total))
            print("Pass/reject test:           %03i ms (%04.1f %%)" % (int(d9 * 1000),  100 * d9 / total))
            print("Clipping:                   %03i ms (%04.1f %%)" % (int(d7 * 1000),  100 * d7 / total))
            print("Perspective divide:         %03i ms (%04.1f %%)" % (int(d8 * 1000),  100 * d8 / total))
            print("Screenspace transform:      %03i ms (%04.1f %%)" % (int(d5 * 1000),  100 * d5 / total))
            print("Line draw:                  %03i ms (%04.1f %%)" % (int(d10 * 1000),  100 * d10 / total))
        """






#### 3D sound ####

import pyaudio
import pydub

CHAN = 8
SAMP = 44100
CLIP = "fitnessgram.mp3"

ANGLES = [-45, 45, 0, 0, -135, 135, -90, 90]
OMNI = [3]

clip = pydub.AudioSegment.from_file(CLIP)
clip = clip.set_channels(1).set_frame_rate(SAMP)
samps = clip.get_array_of_samples()
scale = numpy.iinfo(samps.typecode).max
cliparr = numpy.array(samps).astype(numpy.float32) / scale

angles = numpy.array(ANGLES) * numpy.pi / 180
omnimask = numpy.array([1 if idx in OMNI else 0 for idx in range(CHAN)]) > 0
surmask = ~omnimask

off = 0
ang = 0
dist = 1

def callback(indata, count, tinfo, status):
    global off
    t = numpy.arange(off, off + count) / SAMP
    off += count
    
    #sound = numpy.sin(440 * t * 2 * numpy.pi) * 0.5
    pos = off % cliparr.shape[0]
    sound1 = cliparr[pos:pos + count]
    sound2 = cliparr[:count - sound1.shape[0]]
    sound = numpy.concatenate([sound1, sound2])
    #angle = t * 2 * numpy.pi / 5
    angle = ang + numpy.zeros_like(t)

    dots = numpy.cos(angle[None, :] - angles[:, None])
    amp = numpy.clip(dots, 0, 1) * surmask[:, None]
    amp /= numpy.sum(amp, axis=0)[None, :]
    omni = sound[None, :] * omnimask[:, None]
    sur = sound[None, :] * surmask[:, None] * amp
    audio = (omni + sur) / dist

    outdata = audio.astype(numpy.float32).tobytes("F")
    return (outdata, pyaudio.paContinue)

pa = pyaudio.PyAudio()
stream = pa.open(format=pyaudio.paFloat32,
                 channels=CHAN,
                 rate=SAMP,
                 output=True,
                 stream_callback=callback)

########






lt = time.time()


cube1 = Cube()
cube2 = Cube()
cube3 = Cube()
grid = Grid(101)
#earth = Earth()
objects = [cube1, cube2, cube3, grid]

cube1.position[:] = numpy.array([0, 0.5, 5], dtype=FLOAT_TYPE)
cube2.position[:] = numpy.array([-2, 0.5, 5], dtype=FLOAT_TYPE)
cube3.position[:] = numpy.array([2, 0.5, 5], dtype=FLOAT_TYPE)
grid.scale[:] = numpy.array([50, 1, 50], dtype=FLOAT_TYPE)

camera = Camera()
camera.position[1] = 0.5
camera.position[2] = -2

pygame.init()
clock = pygame.time.Clock()
screen = pygame.display.set_mode(WSIZE)

pygame.mouse.set_visible(False)
pygame.event.set_grab(True)

vel = 4
sens = 1 / 200
tp = time.time()

run = True
while run:
    cam = camera
    td = time.time() - tp
    tp = time.time()
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            run = False

        if event.type == pygame.MOUSEMOTION:
            ang_delta = numpy.array([event.rel[0], -event.rel[1], 0], dtype=FLOAT_TYPE) * sens
            cam.rotation += ang_delta

    if pygame.key.get_pressed()[pygame.K_w]:
        cam.position += Transform().rotate_intrinsic_reverse(cam.rotation).apply(numpy.array([0, 0, vel, 0], dtype=FLOAT_TYPE))[:3] * td
    if pygame.key.get_pressed()[pygame.K_s]:
        cam.position -= Transform().rotate_intrinsic_reverse(cam.rotation).apply(numpy.array([0, 0, vel, 0], dtype=FLOAT_TYPE))[:3] * td
    if pygame.key.get_pressed()[pygame.K_d]:
        cam.position += Transform().rotate_intrinsic_reverse(cam.rotation).apply(numpy.array([vel, 0, 0, 0], dtype=FLOAT_TYPE))[:3] * td
    if pygame.key.get_pressed()[pygame.K_a]:
        cam.position -= Transform().rotate_intrinsic_reverse(cam.rotation).apply(numpy.array([vel, 0, 0, 0], dtype=FLOAT_TYPE))[:3] * td
    if pygame.key.get_pressed()[pygame.K_SPACE]:
        cam.position += Transform().rotate_intrinsic_reverse(cam.rotation).apply(numpy.array([0, vel, 0, 0], dtype=FLOAT_TYPE))[:3] * td
    if pygame.key.get_pressed()[pygame.K_LSHIFT]:
        cam.position -= Transform().rotate_intrinsic_reverse(cam.rotation).apply(numpy.array([0, vel, 0, 0], dtype=FLOAT_TYPE))[:3] * td

    t = time.time()

    cube1.rotation[0] = t * 2 * numpy.pi / 4 % (2 * numpy.pi)
    cube2.position[1] = 0.5 + numpy.sin(t * 2 * numpy.pi / 2)
    cube3.position[2] = 5 + numpy.sin(t * 2 * numpy.pi / 2)

    trans = Transform().translate(-cam.position).rotate_intrinsic(-cam.rotation)
    vec = trans.apply(to_4d(cube1.position[None, :]))[0][:3]
    dist = numpy.linalg.norm(vec)
    ang = numpy.arctan2(vec[0], vec[2])
    
    camera.render(objects, screen)

    pygame.display.flip()
    clock.tick(FPS)

pygame.quit()
stream.close()
pa.terminate()
