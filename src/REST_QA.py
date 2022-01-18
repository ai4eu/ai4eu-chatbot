# FOUNDATION OF RESEARCH AND TECHNOLOGY - HELLAS (FORTH-ICS)
#
# INFORMATION SYSTEMS LABORATORY (ISL)
#
# http://www.ics.forth.gr/isl
#
# LICENCE: TO BE ADDED
#
# Copyright 2021

# The AI4EU chatbot - The REST service for the chatbot

# author: Papadakos Panagiotis
# e-mail: papadako@ics.forth.gr

from sanic import Sanic
from sanic_session import Session, InMemorySessionInterface
from sanic.response import json

from bs4 import BeautifulSoup

import qa.ChatBot_QA as bot

app = Sanic("AI4EU QA chatbot service")

# Manage sessions. Sessions will be alive for 10 minutes
session = Session(app, interface=InMemorySessionInterface(expiry=600, sessioncookie=True))
# Holds the qa chatbot
qa_chatbot = None
# This is the file where we log all question-answer pairs
f = None

"""
"""
def bad_input(error, service="AI4EU QA chatbot"):
    """
    Send a bad service rest answer (400)
    :param service : service name
    :return : json like structure of the error
    """
    return json(
                {
                    'error': 'Wrong input: ' + error,
                    'service': service,
                },
                headers={'X-Served-By': 'AI4EU QA chatbot'},
                status=400)


@app.route('/', methods=["POST"])
async def test(request):
    # We assume we have a json with a query
    data = request.json
    if data is not None:
        query = data['query']
        # Remove HTML escaped characters
        query = str(BeautifulSoup(query, "html.parser"))

        f.write('==>' + query + '\n')
        # interact with the session like a normal dict
        # In the future we will hold here the dialogue state of the user
        if not request.ctx.session.get('ai4eu-session'):
            request.ctx.session['ai4eu-session'] = 'user-dialogue-state-UPDATE'
            print('==> Creating new session!')
        else:
            session_id = test_cookie = request.cookies.get('session')
            print('==> Using existing session with id:' + session_id)

        if query is not None:
            model, results = qa_chatbot.ask(query)
            # Log the top result
            ans, score = results[0]
            f.write('\t' + str(ans) + '\n')
            f.flush()

            return json(
                {
                    'results': results,
                    'model': model,
                    'service': 'AI4EU QA chatbot'
                },
                headers={'X-Served-By': 'AI4EU QA chatbot'},
                status=200)
        else:
            return bad_input('No query field in request body json')

    else:
        return bad_input('No body in request. The body of the post request should be {"query":"the user question"}')


@app.route('/kbqa/', methods=["POST"])
async def test(request):
    # We assume we have a json with a query
    data = request.json
    if data is not None:
        query = data['query']
        # Remove HTML escaped characters
        query = str(BeautifulSoup(query, "html.parser"))

        # interact with the session like a normal dict
        # In the future we will hold here the dialogue state of the user
        if not request.ctx.session.get('ai4eu-session'):
            request.ctx.session['ai4eu-session'] = 'user-dialogue-state-UPDATE'
            print('==> Creating new session!')
        else:
            session_id = test_cookie = request.cookies.get('session')
            print('==> Using existing session with id:' + session_id)

        if query is not None:
            model, results = qa_chatbot.ask_kbqa(query)
            return json(
                {
                    'results': results,
                    'model': model,
                    'service': 'AI4EU QA chatbot'
                },
                headers={'X-Served-By': 'AI4EU QA chatbot'},
                status=200)
        else:
            return bad_input('No query field in request body json')

    else:
        return bad_input('No body in request. The body of the post request should be {"query":"the user question"}')


@app.route('/faq/', methods=["POST"])
async def test(request):
    # We assume we have a json with a query
    data = request.json
    if data is not None:
        query = data['query']
        # Remove HTML escaped characters
        query = str(BeautifulSoup(query, "html.parser"))

        # interact with the session like a normal dict
        # In the future we will hold here the dialogue state of the user
        if not request.ctx.session.get('ai4eu-session'):
            request.ctx.session['ai4eu-session'] = 'user-dialogue-state-UPDATE'
            print('==> Creating new session!')
        else:
            session_id = test_cookie = request.cookies.get('session')
            print('==> Using existing session with id:' + session_id)

        if query is not None:
            model, results = qa_chatbot.ask_faq(query)
            return json(
                {
                    'results': results,
                    'model': model,
                    'service': 'AI4EU QA chatbot'
                },
                headers={'X-Served-By': 'AI4EU QA chatbot'},
                status=200)
        else:
            return bad_input('No query field in request body json')

    else:
        return bad_input('No body in request. The body of the post request should be {"query":"the user question"}')


@app.listener('after_server_start')
def init(sanic, loop):
    """Before starting the service initialize the QA chatbot module"""
    global qa_chatbot
    qa_chatbot = bot.ChatBot_QA()

if __name__ == '__main__':
    # This is a file where we log all queries and the answers
    f = open("queries.txt", "a")
    # Initialized HTML parser
    app.run()
