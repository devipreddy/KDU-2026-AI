# Database Migrations

Run migrations from the `backend` directory:

```powershell
alembic upgrade head
```

Create a future migration after changing SQLAlchemy models:

```powershell
alembic revision --autogenerate -m "describe change"
```
