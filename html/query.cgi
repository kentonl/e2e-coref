#!/usr/bin/env python

import sys
import json
import requests
import cgi, cgitb
cgitb.enable()

SERVER = "http://localhost"
PORT = 10001

def print_header():
  print"Content-Type: text/html"
  print

def print_error(errno, message):
  print "Status: {} {}".format(errno, message)
  print_header()
  sys.exit()

form = cgi.FieldStorage()
if "text" not in form:
  print_error(400, "Bad Request")
text = form.getvalue("text")
try:
  post_response = requests.post(url="{}:{}".format(SERVER, PORT), data={"text":text}, timeout=180)
except requests.exceptions.ConnectionError as e:
  print_error(503, "Service Unavailable")
print_header()
print post_response.text.encode("utf-8")
