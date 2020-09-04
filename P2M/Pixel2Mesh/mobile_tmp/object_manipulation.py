import numpy as np
import re

class Object3D(object):
    """docstring for ClassName"""
    def __init__(self):
        self.vertices=np.array(0)
        self.textures=np.array(0)
        self.faceNormals=np.array(0)
        self.faces=np.array(0)

    def setObj(self, path):
        self.objPath = path

    def readObjetVF(self):
        vertices=[]
        textures=[]
        faces=[]
        normals=[]

        with open(self.objPath, 'r') as f:
            for line in f:
                elts=line.split()
                if elts[0]=='v':
                    vertices.extend([[float(elts[1]),float(elts[2]),float(elts[3])]])
                if elts[0]=='f':
                    elt1=elts[1].split('/')
                    v1=int(elt1[0])
                    elt1=elts[2].split('/')
                    v2=int(elt1[0])
                    elt1=elts[3].split('/')
                    v3=int(elt1[0])
                    faces.extend([[v1,v2,v3]])
                if elts[0]=='vt':
                    textures.extend([[float(elts[1]),float(elts[2])]])


        self.faces=np.asarray(faces)
        self.vertices=np.asarray(vertices)
        self.textures=np.asarray(textures)

    def calculateFaceNormal(self):
        faceNormals=[]
        for f in self.faces:
            (v1,v2,v3)=(np.asarray(self.vertices[f[0]-1]),np.asarray(self.vertices[f[1]-1]),np.asarray(self.vertices[f[2]-1]))
            normal=np.cross(v2-v1,v3-v1)
            faceNormals.extend([normal])
        self.faceNormals=np.asarray(faceNormals)
        return self.faceNormals


    def calculateVerticesNormal(self):
        faceNormals, fIndex=self.faceNormals, 0
        verticesNor=np.array([[0.0,0.0,0.0] for x in self.vertices])
        for f in self.faces:
            fn=faceNormals[fIndex]
            for vf in f:
                vn=verticesNor[vf-1]
                vn[0]=float(vn[0])+fn[0]
                vn[1]=float(vn[0])+fn[1]
                vn[2]=float(vn[0])+fn[2]
                verticesNor[vf-1]=np.asarray(vn)
            fIndex=fIndex+1
        verticesNor=np.asarray(verticesNor)

        for i in range(verticesNor.shape[0]):
            verticesNor[i]=verticesNor[i]/np.linalg.norm(verticesNor[i])
        self.verticesNor=verticesNor
        return self.verticesNor

    def addNormals(self):
        self.readObjetVF()
        self.calculateFaceNormal()
        self.calculateVerticesNormal()

        print(self.vertices.shape)

        vert = np.hstack((np.full([self.vertices.shape[0],1], 'v'), self.vertices))
        face = np.hstack((np.full([self.faces.shape[0],1], 'f'), self.faces))
        #fNormals = np.hstack((np.full([self.faceNormals.shape[0],1], 'fn'), self.faceNormals))
        vNormals = np.hstack((np.full([self.verticesNor.shape[0],1], 'vn'), self.verticesNor))

        interleaved = np.vstack((zip(vNormals,vert)))
        mesh = np.vstack ((interleaved, face))

        np.savetxt(self.objPath, mesh, fmt='%s', delimiter=' ')

        return (self.vertices, self.faces, self.faceNormals, self.textures)
