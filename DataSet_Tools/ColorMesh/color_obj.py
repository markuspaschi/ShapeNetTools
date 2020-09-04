import os,sys,re
import argparse
import numpy as np

def listFiles(dir, ext, ignoreExt=None):
    """
    Return array of all files in dir ending in ext but not ignoreExt.
    """
    matches = []
    for root, dirs, files in os.walk(dir):
        for f in files:
            if f.endswith(ext):
                if not ignoreExt or (ignoreExt and not f.endswith(ignoreExt)):
                    matches.append(os.path.join(root, f))
    return matches

parser = argparse.ArgumentParser(description='Color any obj to a previous colored mesh (vertices have to be the same).')
parser.add_argument('--gt_obj', type=str, help='Ground truth obj', default="data/ellipsoid_four_colors.obj")
parser.add_argument('--objs_folder',  type=str, help='Folder with objs to colorize', default="input")
args = parser.parse_args()

def parse_obj(obj):
    reVertex = re.compile("(?<=^)(v )(.*)(?=$)", re.MULTILINE)
    reVertexN = re.compile("(?<=^)(vn )(.*)(?=$)", re.MULTILINE)
    reFace = re.compile("(?<=^)(f )(.*)(?=$)", re.MULTILINE)

    with open(args.gt_obj) as f:
        lines = f.read()
        vertices = [txt.group() for txt in reVertex.finditer(lines)]
        verticesN = [txt.group() for txt in reVertexN.finditer(lines)]
        faces = [txt.group() for txt in reFace.finditer(lines)]
    return vertices, verticesN, faces

def parse_vertex_colors(vertices):
    verticesRGB = []
    for vertex in vertices:
        reItems = re.compile("([+-]?([0-9]*[.])?[0-9]+)")
        verticesRGB.append([txt.group() for txt in reItems.finditer(vertex)][-3:])
    return verticesRGB

def add_colors(gt_obj, obj):
    pass

gtVertices, _, _ = parse_obj(args.gt_obj)
files = listFiles(args.objs_folder, ".obj")
print(files)
for file in files:

    objVertices, _, _ = parse_obj(file)
    gtVerticesRGB = parse_vertex_colors(gtVertices)

    with open(file, 'r') as f:
        lines = []
        for index, line in enumerate(f):
            if line.startswith("v "):
                line = line.replace("\n", " ") + " ".join(gtVerticesRGB[index]) + "\n"
                lines.append(line)
            else:
                lines.append(line)


    with open(file.replace(".obj", "_colored.obj"), 'w') as f:
        for line in lines:
            f.write(line)
