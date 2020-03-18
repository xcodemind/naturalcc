"""
The flask application package.
"""

from flask import Flask
from flask.ext import restful
from web.webapi.api import HelloWorld

app = Flask(__name__)
api = restful.Api(app)


api.add_resource(HelloWorld, '/testing')