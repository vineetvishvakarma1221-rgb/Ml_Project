import logging
import os 
from datetime import datetime as dt

Log_file=f"{dt.now().strftime('%m_%d_%Y_%M_%S')}.log"
log_path=os.path.join(os.getcwd(),'logs',Log_file)
os.makedirs(log_path,exist_ok=True)

log_file_path=os.path.join(log_path,Log_file)

logging.basicConfig(
    filename=log_file_path,
    format='[ %(asctime)s ]%(lineno)d%(name)s-%(levelname)s-%(message)s',
    level=logging.INFO,
    
)






