import urllib.request
import os
import hashlib
import argparse

models = {
         'undefended':
            {
            'url':
                'https://www.dropbox.com/s/dwefc6wei09lwi0/ResNeXt34_2x32-undefended.pth.tar?dl=1',
             'sha256':
                '38d72d9b381526cbe9eefc0959fc10dad4ff223ed3fd5e60c6ccee363d310912'
            },
        '7step-AT':
            {
            'url':
                'https://www.dropbox.com/s/orm8suhve0njyh7/ResNeXt34_2x32-7step-AT.pth.tar?dl=1',
             'sha256':
                '9c1a529163c04b42822cc2bd4489a838c9ae91af684505b4dbd88edc90dcefae'
            },
        'L1-lambda-0.1':
            {
            'url':
                'https://www.dropbox.com/s/vt590rfyknkev0l/ResNeXt34_2x32-L1-lambda-0.1.pth.tar?dl=1',
            'sha256':
                '71c74e337246dcdf6a6f415ae10cb97c86c107b0fc000f84d9a73831010dc774'
            },
        'L1-lambda-1':
            {
            'url':
                'https://www.dropbox.com/s/tvjgq75bcq0u5si/ResNeXt34_2x32-L1-lambda-1.pth.tar?dl=1',
            'sha256':
                '7bd823def74986a91cf3a35bf80842174fa87f93844ed6036c35c9630d40e9cd'
            },
        'L2-lambda-0.1':
            {
            'url':
                'https://www.dropbox.com/s/cfs3o840eqn02mm/ResNeXt34_2x32-L2-lambda-0.1.pth.tar?dl=1',
            'sha256':
                'a8be9ebb2c374fda5caa464a73fb48d078d574b1c847102cb0e311adcb06bd7f'
            },
        'L2-lambda-1':
            {
            'url':
                'https://www.dropbox.com/s/5wirg57u9rf20ez/ResNeXt34_2x32-L2-lambda-1.pth.tar?dl=1',
            'sha256':
                '723c491e8f2e490931c239465950ea134e061fe095350d8a82e7a0c5a0473e83'
            }
        }


parser = argparse.ArgumentParser('Download pretrained models used in "Scaleable input gradient regularization for adversarial robustness"')

parser.add_argument('model', type=str, choices=models.keys(),metavar='MODEL',
        help='Specify which model to download: "%s"'%'", "'.join(list(models.keys())))
args = parser.parse_args()

model = args.model
fname = models[model]['url'].split('/')[-1].split('?')[0]


os.makedirs('models/pretrained/', exist_ok=True)
path = os.path.join('models/pretrained/',fname)
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
