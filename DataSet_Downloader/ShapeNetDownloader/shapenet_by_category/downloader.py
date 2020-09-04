import os, sys, json, re, ssl, time, requests, logging
from pathlib import Path
import urllib.request, urllib.parse
import numpy as np
import pandas as pd
import utils
from joblib import Parallel, delayed
import subprocess
import warnings

FORMAT = '%(asctime)-15s [%(levelname)s] %(message)s'
logging.basicConfig(format=FORMAT)
log = logging.getLogger('create_subset')
log.setLevel(logging.INFO)
warnings.simplefilter("ignore", UserWarning)

config = {'align': True, 'convert': True, 'normalize': True, 'clean_up': True}

PARALLEL_JOBS = 15 # use 10 for GPU heavy workplaces
BLENDER_DIR = "../../../DataSet_Tools/Renderer/p2m_blender/blender_linux/"

#BLENDER_DIR = "../../../DataSet_Tools/Renderer/p2m_blender/blender_macos/blender.app/Contents/MacOS/"
#PARALLEL_JOBS = 1 # use 1 for macOS -> blender crashes

categories = [
    'key','knife','spoon','cutlery','ruler','tape','measure','Calculator',
    'USB-Stick','CD','DVD','Remote Control','Tongs','Gear','Rubber','Chisel',
    'Scissors','Drill','Bevel','Hammer','Plunger','Screwdriver','Wrench',
    'Spanner','Pencil','Saw','Puncher','Hand Blower','Nail','Flashlight',
    'Light','Bulb','Hand glass','Gloves','Nut','Bolt','Bottle', 'Mallet',
    'Corkscrew', 'Mug'
]

# mesh problems / take forever to process
bad_files = [
    'f33dc3b7728b2d6650b16c6de37deb0b',
    '83cb9de7071ef25a6ef3e946e96cc65c',
    'b8dc144833eed85deceb5f2a99ccd58',
    '7fe9980f9902283ea16aa3a9b1eb42f3',
    '714bcb9410a0cc15b565077beb26776c',
    'f399df73c1e3ff9e3087f84b199fd297',
    '7506da8830c109f626d217fcc203f823',
    'e4a1aebe32a963d1c76869d0300b015f',
    'dba10dd715f350eb598a453fd9fbd988',
    '2910bd7f008fc0c917632fc573934503',
    'cc0f7ae8598a4545ddf3fada108c8396',
    'cfa2a6d21dbdd1891615811ef80dfeea',
    '1a321b2c7192a984ebea1e3786b197cf',
    'c0b6be0fb83b237f504721639e19f609',
    '871a173460e616816c733c74e400f619',
    'ec7cf67cdd3a4d4fafe1d4530f4c6e24',
    '2b7926fc9384ebf09df6f10c48eb6cee',
    '4b2a2a7f5c2376744d041f247323ad5a',
    '6773d093569948e3d40e00f72aac160e',
    'b040cdc4f545282db2fafc90e1c79c5f',
    '63b56cc7aea6a85d2e5d2ccb3226f435',
    '3ad00ae9ce06fcf2f84bda9a0c77bafa',
    '9fa4c43a48b2530ac20423b4affcd786',
    '49c73213ceb3a9405ce473ef7d28f29a',
    '29c4ca047531b76c6cd9fc6cdfdaeb7a',
    'f4351926875ff68d3284cc1e5ee9d86c',
    'f199dee2ca8c56e14cfe6d1b43feb5a2'
]



class Downloader(object):

    def __init__(self, root: str = './'):

        self.root = Path(root)

        self.shapenet_root = self.root / 'generated'
        self.shapenet_object_root = self.shapenet_root / 'meshes'
        self.shapenet_object_root.mkdir(parents=True, exist_ok=True)
        self.model_list_csv = os.path.join(self.shapenet_root, 'full_model_list.csv')

        # has to be done in sub-threads as well!
        self.allow_not_certfied_urls()

        # fetch metadata
        self.get_taxonomy()
        self.get_full_model_list()

        # generate models according to search_list
        models = self.generate_model_list()

        # download models including aligning, normalizing, converting & cleanup
        self.download_required_models(models)

        self.remove_bad_files()

        # removes kmz files and other unimportant stuff
        self.cleanup()

        # create train and test list with absolute path to model.obj files
        # self.generate_train_test_list(models)

    def generate_model_list(self):
        _models = []

        print("\nGenerating Model List...")

        for i, category in enumerate(categories):
            model = self.get_models_by_name(category)
            _models.append(model)

        # Sort the models in descending length order (categories with the most
        # models will drop duplicates)
        _models.sort(key=len, reverse=True)
        models = pd.concat(_models)
        models = models.drop_duplicates(subset=models.columns.difference(['model_name']))

        print(models.groupby(['model_name']).size())
        print("complete dataset length : {}".format(len(models)))

        return models

    """
    def generate_train_test_list(self, models):

        models["model_path"] = models.apply(lambda x:
            self.create_model_path(x['fullId'], x['model_name']), axis=1)

        train_path = os.path.join(self.shapenet_root, 'train_list.txt')
        test_path = os.path.join(self.shapenet_root, 'test_list.txt')

        np.savetxt(train_path, models["model_path"].values, fmt='%s', delimiter="\t")

    def create_model_path(self, fullId, model_name):
        id = utils.getModelId(fullId)
        return str(os.path.abspath(os.path.join(
            self.shapenet_object_root, model_name, id, "models", "model.obj")))
    """

    def download_required_models(self, models):

        # Parallel Download
        with Parallel(n_jobs=PARALLEL_JOBS, verbose=1) as parallel:
            parallel(delayed(self.download_required_model)(model) for index, model in models.iterrows())

        # Align all
        self.align_models()

        # Parallel conversion
        daes = utils.listFiles(self.shapenet_object_root, ext='_aligned.dae')
        with Parallel(n_jobs=PARALLEL_JOBS, verbose=1) as parallel:
            parallel(delayed(self.convert_model)(dae) for dae in daes)

        self.normalize_models()

    def download_required_model(self, model):
        self.allow_not_certfied_urls()

        model_category = model['model_name']
        model_id = model['fullId']
        output_path = os.path.join(self.shapenet_object_root, model_category)
        model_info = utils.getModelInfo(model_id)

        if not (os.path.isdir(output_path + '/' + model_info['id'])):
            utils.downloadKMZ(model_info, output_path)
            utils.extractKMZ(model_info['id'] + ".kmz", output_path)

    def align_models(self):
        if config.get('align'):
            log.info('Aligning DAE...')
            A = utils.getAlignments(self.shapenet_root)
            daes = utils.listFiles(self.shapenet_object_root, ext='.dae', ignoreExt='_aligned.dae')
            for dae in daes:
                utils.align(dae, A)
            log.info('Aligned {} DAE files'.format(str(len(daes))))

    def convert_model(self, dae):
        if config.get('convert'):
            log.info('Converting aligned DAE to OBJ...')

            #out_obj = os.path.splitext(dae)[0] + '.obj'
            out_obj = os.path.join(os.path.dirname(dae), 'model.obj')

            if not os.path.isfile(out_obj):
                blender_dir = os.path.abspath(os.path.join(os.getcwd(), BLENDER_DIR))
                subprocess.call([blender_dir + '/blender --background --python dae_obj.py -- %s %s' % (dae, out_obj)], shell=True, stdout=subprocess.DEVNULL)

            log.info('Converted aligned DAE to OBJ.')

    def normalize_models(self):
        if config.get('normalize'):
            log.info('Normalizing OBJ...')
            objs = utils.listFiles(self.shapenet_object_root, ext='_aligned.obj')
            for obj in objs:
                objfile = obj #os.path.join(output_path, obj)
                base = utils.rchop(objfile, '_aligned.obj')
                objnormfile = base + '.obj'
                utils.normalizeOBJ(objfile, objnormfile)
            log.info('Normalized OBJ.')

    def remove_bad_files(self):
        dirs = utils.listDirs(self.shapenet_object_root, bad_files)
        for dir in dirs:
            utils.removeFolder(dir)

    def cleanup(self):
        if config.get('clean_up'):
            objs = utils.listFiles(self.shapenet_object_root, ext="")
            for obj in objs:
                if(obj.endswith('.dae') or obj.endswith('.kml') or obj.endswith('.kmz')):
                    os.remove(obj)

    def allow_not_certfied_urls(self):
        if (not os.environ.get('PYTHONHTTPSVERIFY', '') and
        getattr(ssl, '_create_unverified_context', None)):
            ssl._create_default_https_context = ssl._create_unverified_context

    def url_encode_string(self, s):
        return urllib.parse.quote(s)

    def get_models_by_name(self, model_name):

        model_list = pd.read_csv(self.model_list_csv, quotechar='"', skipinitialspace=True)
        model_list = model_list[model_list['wnlemmas'].str.contains("(?:(?<![a-zA-Z]| )({})(?![a-zA-Z]| ))".format(model_name), case=False)]

        # dont show keyboard when key is searched for (keyboard,key,...)
        model_list = model_list[~model_list['wnlemmas'].str.contains("({}[a-zA-Z]+)|([a-zA-Z]+{})".format(model_name, model_name), case=False)]
        model_list['model_name'] = model_name.lower().replace(" ", "_")

        return model_list

    def get_full_model_list(self):
        if(os.path.exists(self.model_list_csv)):
            return

        # load once (might take up to one minute) - currently 218403 items
        start = 0
        end = 5000000

        query = "https://www.shapenet.org/solr/models3d/select?q=wnhypersynsets%3Asummer14+AND+source%3A(3dw+OR+yobi3d)&wt=csv&sort=+hasModel+desc%2C+popularity+desc&start={}&rows={}&fq=++%2BhasModel%3Atrue+-modelSize%3A%5B10000000+TO+*+%5D&fl=fullId%2Cwnsynset%2Cwnlemmas%2Cup%2Cfront%2Cname%2Ctags".format(start, end)
        urllib.request.urlretrieve(query, self.model_list_csv, Utils.reporthook)

    def get_taxonomy(self):
        r"""Download the taxonomy from the web."""
        taxonomy_location = os.path.join(self.shapenet_root, 'taxonomy.json')
        if not os.path.exists(taxonomy_location):
            with print_wrapper("Downloading taxonomy ..."):
                taxonomy_web_location = 'http://shapenet.cs.stanford.edu/shapenet/obj-zip/ShapeNetCore.v1/taxonomy.json'
                urllib.request.urlretrieve(taxonomy_web_location,
                                           filename=taxonomy_location)

    def get_category_paths(self, category='chair'):
        r"""Get the list of SynSet IDs and the respective tags based on the taxonomy.

        Args:
            category (str): category of the object that needs to be rtrieved.

        Returns:
            synsetIds (list): list of synsets
            tags (list): list of tags for each synset
        """
        with open(os.path.join(self.shapenet_root, 'taxonomy.json'), 'r') as json_f:
            taxonomy = json.load(json_f)

        synsetIds, children = [], []
        parent_tags, tags = [], []

        for c in taxonomy:
            tag = c['name']

            matchObj = True
            if category is not None:
                matchObj = re.search(
                    r'(?<![a-zA-Z0-9])' + category + '(?![a-zA-Z0-9])', tag,
                    re.M | re.I)

            if matchObj:
                sid = c['synsetId']
                if not sid in synsetIds:
                    synsetIds.append(sid)
                    tags.append(tag)
                for childId in c['children']:
                    if not childId in children:
                        children.append(childId)
                        parent_tags.append(tag)

        while len(children) > 0:
            new_children = []
            new_parent_tags = []
            for c in taxonomy:
                sid = c['synsetId']
                if sid in children and not sid in synsetIds:
                    synsetIds.append(sid)
                    i = children.index(sid)
                    tag = c['name'] + ',' + parent_tags[i]
                    tags.append(tag)
                    for childId in c['children']:
                        if not childId in new_children:
                            new_children.append(childId)
                            new_parent_tags.append(tag)

            children = new_children
            parent_tags = new_parent_tags

        return synsetIds, tags

class Utils(object):
    def reporthook(count, block_size, total_size):

        global start_time
        if count == 0:
            start_time = time.time()
            return
        duration = time.time() - start_time
        progress_size = int(count * block_size)
        speed = int(progress_size / (1024 * duration))
        percent = min(int(count*block_size*100/total_size),100)
        sys.stdout.write("\r...%d%%, %d MB, %d KB/s, %d seconds passed" %
                        (percent, progress_size / (1024 * 1024), speed, duration))
        sys.stdout.flush()


class print_wrapper(object):
    def __init__(self, text, logger=sys.stdout.write):
        self.text = text
        self.logger = logger

    def __enter__(self):
        self.logger(self.text)

    def __exit__(self, *args):
        self.logger("\t[done]\n")

if __name__== "__main__":
    Downloader()
