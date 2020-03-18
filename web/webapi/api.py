from flask import Flask
from flask.ext import restful



class HelloWorld(restful.Resource):
    def get(self):
        return {'test': 'world'}

