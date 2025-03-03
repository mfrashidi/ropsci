"""
ASGI config for ropsci project.

It exposes the ASGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/4.2/howto/deployment/asgi/
"""

import os

from django.core.asgi import get_asgi_application
from channels.routing import ProtocolTypeRouter, URLRouter
from channels.auth import AuthMiddlewareStack
from django.urls import path, re_path

from game import consumers

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ropsci.settings')

application = ProtocolTypeRouter({
    "http": get_asgi_application(),
    "websocket": AuthMiddlewareStack(
        URLRouter([
            re_path(r"^ws/game/(?P<game_id>\w+)/$", consumers.GameConsumer.as_asgi()),
        ])
    ),
})

