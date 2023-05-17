# 模板序号008
# 开发时间 2022/11/20 11:47
import conf.global_settings as settings

class Settings:
    def __init__(self, settings):

        for attr in dir(settings):
            if attr.isupper():
                setattr(self, attr, getattr(settings, attr))

settings = Settings(settings)