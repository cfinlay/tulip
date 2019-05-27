import urllib.request
import os
import hashlib
import argparse

models = {
         'undefended':
            {
            'url':
                 'https://www.dropbox.com/s/d2soq4t72g2fpeb/resnet50-undefended.pth.tar?dl=1',
             'sha256':
                'c57860737bbbc100817ea1c289e35fb27a8aef3be37a2c2f6d20b7267ea00be4'
            },
         'L2-lambda-0.1':
            {
            'url':
                'https://www.dropbox.com/s/3g9bx1m1vzfr1wl/resnet50-L2-lambda-0.1.pth.tar?dl=1',
            'sha256':
                'b63bdb2be8f4eaff21d23d250e751bb57ed8b9189160f46301f416086359ca90'
            },
        'L2-lambda-1':
            {
            'url':
                'https://www.dropbox.com/s/aioksw9silbm0mc/resnet50-L2-lambda-1.pth.tar?dl=1',
            'sha256':
                '4073c3847088e7e4944b9a063bf8ad519ef3aac8e207e06194c68a081a46b992'
            }
        }



parser = argparse.ArgumentParser('Download pretrained models used in "Scaleable input gradient regularization for adversarial robustness"')

parser.add_argument('model', type=str, choices=models.keys(),metavar='MODEL',
        help='Specify which model to download: "%s"'%'", "'.join(list(models.keys())))
args = parser.parse_args()

model = args.model
fname = models[model]['url'].split('/')[-1].split('?')[0]


os.makedirs('pretrained/', exist_ok=True)
path = os.path.join('pretrained/',fname)
if not os.path.isfile(path):
    print('Downloading model to %s'%path)
    urllib.request.urlretrieve(models[model]['url'], path)
    print('Done\n')
else:
    print('Model already downloaded\n')


print('Checking model SHA256 checksum')
sha256 = hashlib.sha256()
with open(path, 'rb') as f:
  data = f.read()
  sha256.update(data)

checkhash = sha256.hexdigest()==models[model]['sha256']
if not checkhash:
    raise ValueError('SHA256 checksum failed!')
else:
    print('Passed')
