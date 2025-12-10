# Docket Viewer Application

## Running the server

You'll need to use Postgres with pgvector installed. If you want to use Docker, you can use the following command:

```bash
docker run -d \
    --name pgvector \
    --env-file .env \
    -p 5432:5432 \
    -v pgdata:/var/lib/postgresql/data \
    pgvector/pgvector:pg16
```

You can run Django management commands with `clx manage`.

For example, to run the development server:

```bash
clx manage runserver
```

If you are starting from scratch, you can initialize the database with:

```bash
clx manage makemigrations app
clx manage migrate
```
