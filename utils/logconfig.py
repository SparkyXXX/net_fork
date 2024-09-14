import logging
import time
import os
import yaml

def log_config(args):
    # 读取保存状态
    save = args.log
    # 读取config文件,以字典的形式存于config
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.load(f, Loader = yaml.FullLoader)

    # 判断是否要生成日志，如果需要生成日志，就读取存储路径，如果不生成，就直接设置显示模式
    # 如果生成日志，则不显示日志
    if save:
        # 读取日志文件的存储路径
        save_path = config['log_params']['logpath']
        #判断存储路径是否存在，如果不存在就创造一个路径
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        # 获取当时的时间，作为log的名字
        ticks = time.asctime(time.localtime(time.time()) )
        ticks = str(ticks).replace(' ', '-').replace(':','-')
        log_name = '{}.log'.format(os.path.join(save_path, ticks))

        logging.basicConfig(format='[%(asctime)s %(levelname)s] %(message)s', 
                            datefmt='%m/%d/%Y %H:%M:%S', 
                            level=logging.DEBUG,
                            filemode='a',
                            filename=log_name)
    else:      
        # 设置代码运行过程中的log信息，在代码调试过程中大家往往用print验证输出，但是大型代码往往需要
        # 记录很多的节点信息，往往这些信息是存入文件供人查看的，实现这个功能的方法就是使用logging模块
        # 在打印输出信息的时候，log完全可以代替print
        # 设置log的输出格式
        logging.basicConfig(format='[%(asctime)s %(levelname)s] %(message)s', 
                            datefmt='%m/%d/%Y %H:%M:%S', 
                            level=logging.DEBUG,)