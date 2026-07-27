-- Per-project long-term memory scope.
-- Existing rows default to global scope (''); new writes record cwd.
ALTER TABLE memories ADD COLUMN cwd TEXT NOT NULL DEFAULT '';
