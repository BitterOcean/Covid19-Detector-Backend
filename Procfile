release: python manage.py migrate --run-syncdb
web: gunicorn backend.wsgi --log-file -
