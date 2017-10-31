#!/usr/bin/env python

import sys
import json
import requests
import cgi, cgitb
cgitb.enable()

SERVER = "http://localhost"
PORT = 10001

form = cgi.FieldStorage()
if "text" not in form:
  print "Status: %d %s\n\n" % (400, "Bad Request")
  sys.exit(0)
text = form.getvalue("text")
try:
  post_response = requests.post(url="{}:{}".format(SERVER, PORT), data={"text":text}, timeout=180)
except requests.exceptions.ConnectionError as e:
  print e
  sys.exit(0)

print "Content-Type: text/html"
print

print post_response.text.encode("utf-8")
