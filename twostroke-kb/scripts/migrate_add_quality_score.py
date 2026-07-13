from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import get_connection


def main() -> None:
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute("ALTER TABLE chunks ADD COLUMN IF NOT EXISTS quality_score FLOAT DEFAULT 0.0")
        cur.execute("ALTER TABLE chunks ADD COLUMN IF NOT EXISTS vote_count INT DEFAULT 0")
        cur.execute("ALTER TABLE chunks ADD COLUMN IF NOT EXISTS downvote_count INT DEFAULT 0")
        cur.execute("CREATE INDEX IF NOT EXISTS chunks_quality_score_idx ON chunks (quality_score)")
        conn.commit()
    finally:
        conn.close()
    print("Migration complete.")


if __name__ == "__main__":
    main()
