from setuptools import setup,find_packages
from typing import List
def get_requirement(file_path:str)-> List[str]:
    req = []
    hypen = '-e .'
    with open(file_path) as f_obj :
        req = f_obj.readlines()
        [r.replace('/n','') for r in req]
        if hypen in req:
            req.remove(hypen)
        

setup(
    name = 'ML_Project',
    version = '0.0.1',
    author = 'Vineet',
    author_email = 'vineetvishvakarma1221@gmail.com',
    packages = find_packages(),
    install_requires = get_requirement('requirements.txt')
 

)



