"""Build a six-region sphere from the repository's unchanged RSLCube textures."""
import json
import math
from pathlib import Path
import shutil
import struct
import zipfile
from PIL import Image
import numpy as np

OUT = Path(__file__).resolve().parent
R = 0.0325
N = 64
# Each u x v points outwards; the six spherical patches are congruent.
BASES = [((1,0,0),(0,1,0),(0,0,1)),((-1,0,0),(0,-1,0),(0,0,1)),
         ((0,1,0),(-1,0,0),(0,0,1)),((0,-1,0),(1,0,0),(0,0,1)),
         ((0,0,1),(1,0,0),(0,1,0)),((0,0,-1),(1,0,0),(0,-1,0))]
NAMES = ['posX','negX','posY','negY','posZ','negZ']
def dot(a,b): return sum(x*y for x,y in zip(a,b))
def unit(a):
    length=math.sqrt(dot(a,a))
    return tuple(x/length for x in a)
def vectors(values): return '['+', '.join('('+', '.join(f'{x:.9g}' for x in p)+')' for p in values)+']'

(OUT/'textures').mkdir(exist_ok=True, parents=True)
# Cube-projected spherical caps, matching the 38-degree printable sphere.
size = 1024
y, x = np.mgrid[:size, :size]
a = 2 * (x + 0.5) / size - 1
b = 1 - 2 * (y + 0.5) / size
cap_radius = math.tan(math.radians(38))
inside = a*a + b*b <= cap_radius*cap_radius
for i in range(6):
    original = np.asarray(Image.open(OUT / f'source_textures/texture{i+1}.png').convert('RGB'))
    tx = np.clip(np.rint((0.5+a/(2*cap_radius))*(original.shape[1]-1)).astype(int), 0, original.shape[1]-1)
    ty = np.clip(np.rint((0.5-b/(2*cap_radius))*(original.shape[0]-1)).astype(int), 0, original.shape[0]-1)
    pixels = np.empty((size,size,3), dtype=np.uint8)
    pixels[:] = (63,149,157)
    pixels[inside] = original[ty[inside],tx[inside]]
    Image.fromarray(pixels).save(OUT / f'textures/texture{i+1}.png')
blob=bytearray()
gltf={'asset':{'version':'2.0','generator':'ViserDex six-region sphere'},'scene':0,
      'scenes':[{'nodes':[0]}],'nodes':[{'mesh':0,'name':'RSLSphere65'}],
      'meshes':[{'primitives':[]}],'buffers':[],'bufferViews':[],'accessors':[],
      'images':[],'textures':[],'materials':[], 'samplers':[{'magFilter':9729,'minFilter':9987,'wrapS':33071,'wrapT':33071}]}
def view(data):
    blob.extend(b'\0'*((-len(blob))%4))
    idx=len(gltf['bufferViews'])
    gltf['bufferViews'].append({'buffer':0,'byteOffset':len(blob),'byteLength':len(data)})
    blob.extend(data)
    return idx
def accessor(rows,typ,component=5126,bounds=False):
    flat=[x for row in rows for x in row]
    v=view(struct.pack('<'+('f' if component==5126 else 'I')*len(flat),*flat))
    a={'bufferView':v,'componentType':component,'count':len(rows),'type':typ}
    if bounds:
        a.update(min=[min(col) for col in zip(*rows)],max=[max(col) for col in zip(*rows)])
    gltf['accessors'].append(a)
    return len(gltf['accessors'])-1
usd=['#usda 1.0','(defaultPrim = "RSLSphere65"\n metersPerUnit = 1\n upAxis = "Z")',
     'def Xform "RSLSphere65" (prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]) {',
     ' float physics:mass = 0.07', ' point3f physics:centerOfMass = (0, 0, 0)',
     f' float3 physics:diagonalInertia = ({0.4*.07*R*R}, {0.4*.07*R*R}, {0.4*.07*R*R})',
     ' def Sphere "collision" (prepend apiSchemas = ["PhysicsCollisionAPI", "MaterialBindingAPI"]) {',
     f'  double radius = {R}\n  token visibility = "invisible"\n  bool physics:collisionEnabled = true\n  rel material:binding:physics = </RSLSphere65/ContactMaterial>\n }}',
     ''' def Material "ContactMaterial" (prepend apiSchemas = ["PhysicsMaterialAPI"]) {
 float physics:staticFriction = 0.8
 float physics:dynamicFriction = 0.6
 float physics:restitution = 0
 }''', ' def Scope "Looks" {']
mesh_parts=[]
for k,(normal,u,v) in enumerate(BASES):
    pos=[]; norms=[]; uv=[]; faces=[]
    for j in range(N+1):
        for i in range(N+1):
            a=2*i/N-1; b=2*j/N-1
            n=unit([normal[d]+a*u[d]+b*v[d] for d in range(3)])
            pos.append(tuple(R*x for x in n)); norms.append(n); uv.append((i/N,j/N))
    for j in range(N):
        for i in range(N):
            a=j*(N+1)+i; b=a+1; c=a+N+1; d=c+1
            faces.extend([(a,b,d),(a,d,c)])
    attrs={'POSITION':accessor(pos,'VEC3',bounds=True),'NORMAL':accessor(norms,'VEC3'),
           'TEXCOORD_0':accessor([(s,1-t) for s,t in uv],'VEC2')}
    inds=accessor([(x,) for face in faces for x in face],'SCALAR',5125)
    gltf['meshes'][0]['primitives'].append({'attributes':attrs,'indices':inds,'material':k})
    gltf['images'].append({'bufferView':view((OUT/f'textures/texture{k+1}.png').read_bytes()),'mimeType':'image/png'})
    gltf['textures'].append({'source':k,'sampler':0})
    gltf['materials'].append({'name':NAMES[k],'pbrMetallicRoughness':{'baseColorTexture':{'index':k},'metallicFactor':0,'roughnessFactor':.75}})
    path=f'/RSLSphere65/Looks/face{k}'
    usd.append(f''' def Material "face{k}" {{
 token outputs:surface.connect = <{path}/surface.outputs:surface>
 def Shader "surface" {{
  uniform token info:id = "UsdPreviewSurface"
  color3f inputs:diffuseColor.connect = <{path}/texture.outputs:rgb>
  float inputs:roughness = 0.75
  token outputs:surface
 }}
 def Shader "texture" {{
  uniform token info:id = "UsdUVTexture"
  asset inputs:file = @textures/texture{k+1}.png@
  token inputs:sourceColorSpace = "sRGB"
  token inputs:wrapS = "clamp"
  token inputs:wrapT = "clamp"
  float2 inputs:st.connect = <{path}/uv.outputs:result>
  float3 outputs:rgb
 }}
 def Shader "uv" {{
  uniform token info:id = "UsdPrimvarReader_float2"
  token inputs:varname = "st"
  float2 outputs:result
 }}
 }}''')
    mesh_parts.append(f''' def Mesh "{NAMES[k]}" (prepend apiSchemas = ["MaterialBindingAPI"]) {{
  point3f[] points = {vectors(pos)}
  int[] faceVertexCounts = {json.dumps([3]*len(faces))}
  int[] faceVertexIndices = {json.dumps([x for f in faces for x in f])}
  normal3f[] normals = {vectors(norms)} (interpolation = "vertex")
  texCoord2f[] primvars:st = {vectors(uv)} (interpolation = "vertex")
  uniform token subdivisionScheme = "none"
  rel material:binding = <{path}>
 }}''')
usd+=[' }',*mesh_parts,'}']
(OUT/'RSLSphere65.usda').write_text('\n'.join(usd))
gltf['buffers']=[{'byteLength':len(blob)}]
js=json.dumps(gltf,separators=(',',':')).encode(); js+=b' '*((-len(js))%4)
blob.extend(b'\0'*((-len(blob))%4))
(OUT/'RSLSphere65.glb').write_bytes(struct.pack('<III',0x46546c67,2,28+len(js)+len(blob))+struct.pack('<II',len(js),0x4e4f534a)+js+struct.pack('<II',len(blob),0x004e4942)+blob)

metadata = {
    "name": "RSLSphere65", "diameter_m": 2*R, "mass_kg": 0.07,
    "mass_source": "70 g explicitly selected by user; not a measured value",
    "inertia_model": "uniform solid sphere", "diagonal_inertia_kg_m2": [0.4*0.07*R*R]*3,
    "background_rgb": [63,149,157], "cap_half_angle_deg":38,
    "triangles": 6*N*N*2,
    "regions": [{"axis":name,"texture":f"textures/texture{i+1}.png","letter":letter}
                for i,(name,letter) in enumerate(zip(NAMES,["E","T","H","R","S","L"]))],
    "collision": "one analytic USD Sphere, radius 0.0325 m",
    "contact_material": {"static_friction":0.8,"dynamic_friction":0.6,"restitution":0.0,
                         "status":"initial simulation values, not measured"},
    "coordinate_system": "USD: Z-up, right-handed, meters; GLB vertex coordinates retained",
    "limitations": ["Ideal sphere: no physical grooves or print seams modeled.",
                    "No Gaussian splat or pose network supplied.",
                    "Runtime Newton import/contact validation not performed.",
                    "GLB is a visual asset; physics is defined in USDA."]
}
(OUT/'asset.json').write_text(json.dumps(metadata,indent=2))
print(OUT/'RSLSphere65.usda')
