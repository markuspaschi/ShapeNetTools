import csv
import json
import numpy as np
import os
import shutil
import urllib
from subprocess import call
from multiprocessing import Pool
import zipfile

SERVER = 'https://www.shapenet.org/'
DATA_URL = SERVER + '/shapenet/data/'
SOLR_URL = SERVER + '/models3d/solr/select'
SCREENSHOTS_URL = SERVER + '/shapenet/screenshots/models/3dw/'


def getModelList(dataset='ShapeNetCore', synsetId='*'):
    """
    Return array of model full ids in given synsetId and dataset.
    """
    encodedQuery = 'datasets:%s AND wnhypersynsets:%s' % (dataset, synsetId)
    url = '{}?q={}&rows=10000000&fl=fullId&wt=csv&csv.header=false'
    solrQueryURL = url.format(SOLR_URL, encodedQuery)
    response = urllib.urlopen(solrQueryURL)
    return response.read().splitlines()


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

def listDirs(dir, names):
    """
    Return array of all files in dir ending in ext but not ignoreExt.
    """
    matches = []
    for root, dirs, files in os.walk(dir):
        for dir in dirs:
            if dir in names:
                matches.append(os.path.join(root, dir))
    return matches

# TODO: make less naive
def getIdFromPath(path):
    """
    Return model id from path
    """
    parts = []
    (path, tail) = os.path.split(path)
    while path and tail:
        parts.append(tail)
        (path, tail) = os.path.split(path)
    parts.append(os.path.join(path, tail))
    res = list(map(os.path.normpath, parts))[::-1]
    res = [k for k in res if len(k) > 28]
    return res[0] if len(res) > 0 else ''


def getPrefixedPath(prefixLength, id):
    """
    Returns a path with format id[0]/id[1]/.../id[prefixLength]/id/
    """
    prefix = id[:prefixLength]
    rest = id[prefixLength:]
    path = ''
    for i in range(0, prefixLength):
        path = path + prefix[i] + '/'
    path = path + rest + '/' + id + '/'
    return path


def getModelInfo(fullId):
    """
    Returns an info object with URLs to model resources.
    """
    info = {}
    srcId = fullId.split('.')
    id = srcId[1]
    info['source'] = srcId[0]
    info['id'] = id
    prefix = getPrefixedPath(5, id)
    info['kmz'] = DATA_URL + prefix + 'Collada/' + id + '.kmz'
    info['png'] = SCREENSHOTS_URL + prefix + id + '-%i.png'
    info['gif'] = SCREENSHOTS_URL + prefix + id + '.gif'
    return info

def getModelId(fullId):
    srcId = fullId.split('.')
    return srcId[1]

def downloadPNGs(fullId, outDir):
    """
    Downloads PNG screenshots for model corresponding to info into outDir
    """
    if not os.path.isdir(outDir):
        os.makedirs(outDir)
    info = getModelInfo(fullId)
    for i in range(14):
        pngUrl = info['png'] % i
        imgSuffix = info['id'] + '-' + str(i) + '.png'
        localFile = os.path.join(outDir, imgSuffix)
        if not os.path.isfile(localFile):
            urllib.request.urlretrieve(pngUrl, localFile)
            print (pngUrl)


def downloadKMZ(info, outDir):
    """
    Downloads KMZ model corresponding to info into outDir/id.kmz
    """
    localFile = outDir + '/' + info['id'] + '.kmz'

    if not os.path.isdir(outDir):
        try:
            os.makedirs(outDir)
        except:
            pass

    if not os.path.isfile(localFile):
        url = info['kmz']
        urllib.request.urlretrieve(url, localFile)
        print (url)


def downloadKMZs(synsetId, outDir):
    """
    Downloads all KMZ model files in synsetId to outDir/synsetId.
    """
    if not os.path.isdir(outDir):
        os.makedirs(outDir)
    modelIds = getModelList(dataset='ShapeNetCore', synsetId=synsetId)
    for fullId in modelIds:
        info = getModelInfo(fullId)
        downloadKMZ(info, outDir)

def extractKMZ(kmzFile, kmzDir):
    """
    Extracts single KMZ file
    """
    subdir = os.path.join(kmzDir, os.path.splitext(kmzFile)[0])
    if not os.path.isdir(subdir):
        inputFile = os.path.join(kmzDir, kmzFile)
        print(inputFile + '\t->\t' + subdir)
        zf = zipfile.ZipFile(inputFile, 'r')
        zf.extractall(subdir)

def extractKMZs(kmzDir):
    """
    Extracts all KMZ files in kmzDir to directories with basename of each.
    """
    for f in os.listdir(kmzDir):
        if f.endswith('.kmz'):
            subdir = os.path.join(kmzDir, os.path.splitext(f)[0])
            if not os.path.isdir(subdir):
                inputFile = os.path.join(kmzDir, f)
                print(inputFile + '\t->\t' + subdir)
                zf = zipfile.ZipFile(inputFile, 'r')
                zf.extractall(subdir)


def dae2obj(dae):
    obj = os.path.splitext(dae)[0] + '.obj'
    if not os.path.isfile(obj):
        call(['assimp', 'export', dae, obj])

def getAlignments(output_path):
    """
    Returns dic from model id to alignment matrix as row-major string.
    """
    alignmentFile = os.path.join(output_path, 'ShapeNetCore-alignments.csv')
    alignURL = ('https://www.shapenet.org/solr/models3d/select?q='
                'datasets%3AShapeNetCore&fl=id%2Cup%2Cfront'
                '&rows=100000&wt=csv&csv.header=false')
    if not os.path.isfile(alignmentFile):
        print('Downloading alignments from ' + alignURL + '...')
        urllib.request.urlretrieve(alignURL, alignmentFile)
    alignments = {}
    with open(alignmentFile) as upFrontFile:
        # print('Reading alignments from ' + alignmentFile + '...')
        reader = csv.reader(upFrontFile)
        alignments = {}
        for line in reader:
            modelId = line[0]
            up = np.fromstring(line[1], sep='\\,')  # up is y
            fr = np.fromstring(line[2], sep='\\,')  # front is -z
            x = np.cross(fr, up)  # x is cross product (and also right)
            y = up
            z = np.negative(fr)
            mat4rowwise = np.concatenate(
                [x, [0], y, [0], z, [0], [0, 0, 0, 1]])
            alignments[modelId] = ' '.join([str(num) for num in mat4rowwise])
    return alignments


def align(dae, alignments):
    """
    Aligns given .dae file and writes result to _aligned.dae
    """
    daeAligned = os.path.join(os.path.dirname(dae),'model_aligned.dae')

    #daeAligned = os.path.splitext(dae)[0] + '_aligned.dae'
    if os.path.isfile(daeAligned):
        # print('Skipping alread aligned: ' + daeAligned)
        return
    id = getIdFromPath(dae)
    if id in alignments:
        matrixVals = alignments[id]
        matrixString = '<matrix>' + matrixVals + '</matrix>\n'
        # first part (before ;) replaces up_axis, second part inserts
        # transform in scene
        regex = ('s/<up_axis>.*<\/up_axis>/<up_axis>Y_UP<\/up_axis>/g; '
                 '/<visual_scene/a\\\n' + matrixString)
        f = open(daeAligned, 'w')
        call(['sed', regex, dae], stdout=f)
        print(dae + ' -> ' + daeAligned)
    else:  # just copy file
        print('[WARN] No alignment for %s, so just copying original' % id)
        shutil.copy2(dae, daeAligned)


def binvox(obj):
    """
    Computes surface and solid voxelizations in binvox format for given obj
    """
    binvoxCmd = ['binvox', '-aw', '-dc', '-down', '-pb']
    basename = os.path.splitext(obj)[0]
    binvoxOut = basename + '.binvox'
    binvoxSolid = basename + '.solid.binvox'
    if not os.path.isfile(binvoxSolid):
        call(binvoxCmd + [obj])
        os.rename(binvoxOut, binvoxSolid)
    binvoxSurface = basename + '.surface.binvox'
    if not os.path.isfile(binvoxSurface):
        call(binvoxCmd + ['-ri'] + [obj])
        os.rename(binvoxOut, binvoxSurface)


def binvoxDir(inDir, objExt='.obj'):
    """
    Computes surface and solid binvox voxelizations for objExt in inDir
    """
    objs = listFiles(inDir, ext=objExt)
    pool = Pool()
    pool.map(binvox, objs)
    return


def obj2stats(obj):
    """
    Computes statistics of OBJ vertices and returns as {num,min,max,centroid}
    """
    minVertex = np.array([np.Infinity, np.Infinity, np.Infinity])
    maxVertex = np.array([-np.Infinity, -np.Infinity, -np.Infinity])
    aggVertices = np.zeros(3)
    numVertices = 0
    with open(obj, 'r') as f:
        for line in f:
            if line.startswith('v '):
                v = np.fromstring(line[2:], sep=' ')
                aggVertices += v
                numVertices += 1
                minVertex = np.minimum(v, minVertex)
                maxVertex = np.maximum(v, maxVertex)
    centroid = aggVertices / numVertices
    info = {}
    info['id'] = getIdFromPath(obj)
    info['numVertices'] = numVertices
    info['min'] = minVertex
    info['max'] = maxVertex
    info['centroid'] = centroid
    return info


# Handles serialization of 1D numpy arrays to JSON
class NumpyAwareJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray) and obj.ndim == 1:
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def rchop(thestring, ending):
    """
    Removes ending from thestring if matching
    """
    if thestring.endswith(ending):
        return thestring[:-len(ending)]
    return thestring


def normalizeOBJ(obj, out, stats=None):
    """
    Normalizes OBJ to be centered at origin and fit in unit cube
    """
    if os.path.isfile(out):
        return
    if not stats:
        stats = obj2stats(obj)
    diag = stats['max'] - stats['min']
    norm = 1 / np.linalg.norm(diag)
    c = stats['centroid']
    outmtl = os.path.splitext(out)[0] + '.mtl'
    with open(obj, 'r') as f, open(out, 'w') as fo:
        for line in f:
            if line.startswith('v '):
                v = np.fromstring(line[2:], sep=' ')
                vNorm = (v - c) * norm
                vNormString = 'v %f %f %f\n' % (vNorm[0], vNorm[1], vNorm[2])
                fo.write(vNormString)
            elif line.startswith('mtllib '):
                fo.write('mtllib ' + os.path.basename(outmtl) + '\n')
            else:
                fo.write(line)
    outStats = open(os.path.splitext(out)[0] + '.json', 'w')
    j = json.dumps(stats, cls=NumpyAwareJSONEncoder)
    outStats.write(j + '\n')
    outStats.close()
    #shutil.copy2(obj + '.mtl', outmtl)
    shutil.copy2(obj.replace(".obj", ".mtl"), outmtl)
    return stats


def zipdir(path, ziph, basepath=None, ext=None, ignoreExt=None):
    """
    Recursively adds to ziph files in path matching ext but not ignoreExt.
    Files are added into ziph with relative path from basepath
    """
    if not basepath:
        basepath = path
    for root, dirs, files in os.walk(path):
        for f in files:
            if not ext or f.endswith(ext):
                if not ignoreExt or f.endswith(ignoreExt):
                    fpath = os.path.join(root, f)
                    ziph.write(fpath, os.path.relpath(fpath, basepath))


def zipModelFiles(dir, basepath=None):
    MODELPKG_BASENAME = 'model_normalized'
    MODELPKG_EXTS = ['.obj', '.mtl', '.solid.binvox', '.surface.binvox',
                     '.json']
    COMPRESSION = zipfile.ZIP_DEFLATED

    if not basepath:
        basepath = os.path.normpath(dir + '/../')

    mzip = os.path.normpath(dir) + '.models.zip'
    szip = os.path.normpath(dir) + '.screenshots.zip'
    if os.path.isfile(mzip) and os.path.isfile(szip):
        return

    modeldirs = [d for d in os.listdir(dir)
                 if os.path.isdir(os.path.join(dir, d))]

    with zipfile.ZipFile(mzip, mode='w', compression=COMPRESSION) as zm:
        for id in modeldirs:
            mdir = os.path.join(dir, id)
            for ext in MODELPKG_EXTS:
                f = os.path.join(mdir, 'models', MODELPKG_BASENAME + ext)
                zm.write(f, os.path.relpath(f, basepath))
            zipdir(mdir + '/images', zm, basepath)

    with zipfile.ZipFile(szip, mode='w', compression=COMPRESSION) as zs:
        for id in modeldirs:
            sdir = os.path.join(dir, id, 'screenshots')
            zipdir(sdir, zs, basepath)

def removeFolder(dir):
    shutil.rmtree(dir, ignore_errors=True)
