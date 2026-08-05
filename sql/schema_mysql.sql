-- ---------------------------------------------------------------------------
-- IPL Analytics warehouse - MySQL / MariaDB schema
--
-- The application creates this automatically via SQLAlchemy
-- (`python scripts/init_db.py`); this file is the reference version.
--
--   mysql -u root -p < sql/schema_mysql.sql
--
-- Notes on the MySQL dialect:
--   * utf8mb4 throughout - player names contain non-ASCII characters.
--   * VARCHAR lengths are explicit because MySQL cannot index unbounded TEXT.
--   * BOOLEAN is an alias for TINYINT(1); the application reads 0/1 back as bool.
-- ---------------------------------------------------------------------------

CREATE DATABASE IF NOT EXISTS ipl
    CHARACTER SET utf8mb4
    COLLATE utf8mb4_unicode_ci;

USE ipl;

-- ===========================================================================
-- Dimensions
-- ===========================================================================
CREATE TABLE teams (
    id            INT AUTO_INCREMENT PRIMARY KEY,
    name          VARCHAR(80)  NOT NULL,
    short_code    VARCHAR(10),
    is_active     BOOLEAN      NOT NULL DEFAULT FALSE,
    primary_color VARCHAR(16),
    logo_url      VARCHAR(400),
    created_at    TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY uq_teams_name (name)
) ENGINE=InnoDB;

CREATE TABLE venues (
    id         INT AUTO_INCREMENT PRIMARY KEY,
    name       VARCHAR(160) NOT NULL,
    city       VARCHAR(80),
    country    VARCHAR(60) DEFAULT 'India',
    created_at TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY uq_venues_name (name)
) ENGINE=InnoDB;

CREATE TABLE players (
    id               INT AUTO_INCREMENT PRIMARY KEY,
    name             VARCHAR(120) NOT NULL,
    source_player_id VARCHAR(80),
    short_name       VARCHAR(120),
    batting_style    VARCHAR(60),
    bowling_style    VARCHAR(80),
    role             VARCHAR(40),
    is_overseas      BOOLEAN,
    image_url        VARCHAR(400),
    created_at       TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY uq_players_name (name),
    KEY ix_players_source_id (source_player_id)
) ENGINE=InnoDB;

-- ===========================================================================
-- Facts
-- ===========================================================================
CREATE TABLE matches (
    id                      INT AUTO_INCREMENT PRIMARY KEY,

    match_key               VARCHAR(60) NOT NULL,
    source                  VARCHAR(20) NOT NULL,
    source_match_id         VARCHAR(40),
    competition_id          INT,

    season                  INT NOT NULL,
    match_date              DATE,
    start_datetime          DATETIME,
    match_number            VARCHAR(40),
    stage                   VARCHAR(40),
    is_playoff              BOOLEAN NOT NULL DEFAULT FALSE,
    is_day_night            BOOLEAN,
    overs_per_innings       INT DEFAULT 20,

    venue_id                INT,
    city                    VARCHAR(80),
    is_neutral_venue        BOOLEAN NOT NULL DEFAULT FALSE,

    team1_id                INT,
    team2_id                INT,
    home_team_id            INT,
    away_team_id            INT,

    toss_winner_id          INT,
    toss_decision           VARCHAR(10),

    first_batting_team_id   INT,
    second_batting_team_id  INT,

    is_completed            BOOLEAN NOT NULL DEFAULT FALSE,
    winner_id               INT,
    result_type             VARCHAR(20),
    win_margin_runs         INT,
    win_margin_wickets      INT,
    is_tie                  BOOLEAN NOT NULL DEFAULT FALSE,
    is_no_result            BOOLEAN NOT NULL DEFAULT FALSE,
    is_super_over           BOOLEAN NOT NULL DEFAULT FALSE,
    is_dls_applied          BOOLEAN NOT NULL DEFAULT FALSE,
    target_runs             INT,
    player_of_match_id      INT,
    result_summary          VARCHAR(300),

    umpire1                 VARCHAR(80),
    umpire2                 VARCHAR(80),
    third_umpire            VARCHAR(80),
    match_referee           VARCHAR(80),

    ingested_at             TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at              TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                                ON UPDATE CURRENT_TIMESTAMP,

    UNIQUE KEY uq_matches_key (match_key),
    KEY ix_matches_season      (season),
    KEY ix_matches_date        (match_date),
    KEY ix_matches_completed   (is_completed),
    KEY ix_matches_season_date (season, match_date),
    KEY ix_matches_teams       (team1_id, team2_id),
    KEY ix_matches_winner      (winner_id),
    KEY ix_matches_venue       (venue_id),

    CONSTRAINT fk_matches_venue   FOREIGN KEY (venue_id)              REFERENCES venues (id),
    CONSTRAINT fk_matches_team1   FOREIGN KEY (team1_id)              REFERENCES teams (id),
    CONSTRAINT fk_matches_team2   FOREIGN KEY (team2_id)              REFERENCES teams (id),
    CONSTRAINT fk_matches_home    FOREIGN KEY (home_team_id)          REFERENCES teams (id),
    CONSTRAINT fk_matches_away    FOREIGN KEY (away_team_id)          REFERENCES teams (id),
    CONSTRAINT fk_matches_toss    FOREIGN KEY (toss_winner_id)        REFERENCES teams (id),
    CONSTRAINT fk_matches_first   FOREIGN KEY (first_batting_team_id) REFERENCES teams (id),
    CONSTRAINT fk_matches_second  FOREIGN KEY (second_batting_team_id) REFERENCES teams (id),
    CONSTRAINT fk_matches_winner  FOREIGN KEY (winner_id)             REFERENCES teams (id),
    CONSTRAINT fk_matches_pom     FOREIGN KEY (player_of_match_id)    REFERENCES players (id),

    CONSTRAINT ck_matches_toss_decision
        CHECK (toss_decision IS NULL OR toss_decision IN ('bat', 'field'))
) ENGINE=InnoDB;

CREATE TABLE innings (
    id                 INT AUTO_INCREMENT PRIMARY KEY,
    match_id           INT NOT NULL,
    innings_no         INT NOT NULL,
    batting_team_id    INT,
    bowling_team_id    INT,

    runs               INT,
    wickets            INT,
    overs              FLOAT,
    balls              INT,
    run_rate           FLOAT,

    extras             INT,
    byes               INT,
    leg_byes           INT,
    wides              INT,
    no_balls           INT,
    penalty            INT,

    powerplay_runs     INT,
    powerplay_wickets  INT,
    middle_runs        INT,
    middle_wickets     INT,
    death_runs         INT,
    death_wickets      INT,

    fours              INT,
    sixes              INT,
    dot_balls          INT,
    target             INT,
    is_declared        BOOLEAN NOT NULL DEFAULT FALSE,

    UNIQUE KEY uq_innings_match_no (match_id, innings_no),
    KEY ix_innings_match (match_id),
    CONSTRAINT fk_innings_match FOREIGN KEY (match_id) REFERENCES matches (id) ON DELETE CASCADE,
    CONSTRAINT fk_innings_bat   FOREIGN KEY (batting_team_id) REFERENCES teams (id),
    CONSTRAINT fk_innings_bowl  FOREIGN KEY (bowling_team_id) REFERENCES teams (id),
    CONSTRAINT ck_innings_wickets CHECK (wickets IS NULL OR (wickets >= 0 AND wickets <= 10))
) ENGINE=InnoDB;

CREATE TABLE batting_cards (
    id                INT AUTO_INCREMENT PRIMARY KEY,
    match_id          INT NOT NULL,
    innings_no        INT NOT NULL,
    team_id           INT,
    player_id         INT NOT NULL,

    batting_position  INT,
    runs              INT DEFAULT 0,
    balls             INT DEFAULT 0,
    fours             INT DEFAULT 0,
    sixes             INT DEFAULT 0,
    dot_balls         INT,
    strike_rate       FLOAT,

    is_out            BOOLEAN NOT NULL DEFAULT FALSE,
    dismissal_kind    VARCHAR(40),
    dismissal_text    VARCHAR(200),
    bowler_id         INT,
    fielder_id        INT,
    wicket_number     INT,

    UNIQUE KEY uq_batting_card (match_id, innings_no, player_id),
    KEY ix_batting_match  (match_id),
    KEY ix_batting_player (player_id),
    CONSTRAINT fk_batting_match  FOREIGN KEY (match_id)  REFERENCES matches (id) ON DELETE CASCADE,
    CONSTRAINT fk_batting_player FOREIGN KEY (player_id) REFERENCES players (id),
    CONSTRAINT fk_batting_bowler FOREIGN KEY (bowler_id) REFERENCES players (id),
    CONSTRAINT fk_batting_field  FOREIGN KEY (fielder_id) REFERENCES players (id),
    CONSTRAINT fk_batting_team   FOREIGN KEY (team_id)   REFERENCES teams (id)
) ENGINE=InnoDB;

CREATE TABLE bowling_cards (
    id             INT AUTO_INCREMENT PRIMARY KEY,
    match_id       INT NOT NULL,
    innings_no     INT NOT NULL,
    team_id        INT,
    player_id      INT NOT NULL,

    bowling_order  INT,
    overs          FLOAT DEFAULT 0,
    balls          INT,
    maidens        INT DEFAULT 0,
    runs_conceded  INT DEFAULT 0,
    wickets        INT DEFAULT 0,
    wides          INT DEFAULT 0,
    no_balls       INT DEFAULT 0,
    dot_balls      INT,
    economy        FLOAT,

    UNIQUE KEY uq_bowling_card (match_id, innings_no, player_id),
    KEY ix_bowling_match  (match_id),
    KEY ix_bowling_player (player_id),
    CONSTRAINT fk_bowling_match  FOREIGN KEY (match_id)  REFERENCES matches (id) ON DELETE CASCADE,
    CONSTRAINT fk_bowling_player FOREIGN KEY (player_id) REFERENCES players (id),
    CONSTRAINT fk_bowling_team   FOREIGN KEY (team_id)   REFERENCES teams (id)
) ENGINE=InnoDB;

CREATE TABLE fall_of_wickets (
    id          INT AUTO_INCREMENT PRIMARY KEY,
    match_id    INT NOT NULL,
    innings_no  INT NOT NULL,
    wicket_no   INT NOT NULL,
    player_id   INT,
    team_id     INT,
    fall_score  INT,
    fall_overs  FLOAT,

    UNIQUE KEY uq_fow (match_id, innings_no, wicket_no),
    KEY ix_fow_match (match_id),
    CONSTRAINT fk_fow_match  FOREIGN KEY (match_id)  REFERENCES matches (id) ON DELETE CASCADE,
    CONSTRAINT fk_fow_player FOREIGN KEY (player_id) REFERENCES players (id),
    CONSTRAINT fk_fow_team   FOREIGN KEY (team_id)   REFERENCES teams (id)
) ENGINE=InnoDB;

CREATE TABLE partnerships (
    id                 INT AUTO_INCREMENT PRIMARY KEY,
    match_id           INT NOT NULL,
    innings_no         INT NOT NULL,
    wicket_no          INT NOT NULL,
    team_id            INT,

    striker_id         INT,
    non_striker_id     INT,
    runs               INT,
    balls              INT,
    striker_runs       INT,
    striker_balls      INT,
    non_striker_runs   INT,
    non_striker_balls  INT,
    extras             INT,
    start_over         FLOAT,
    end_over           FLOAT,
    is_unbroken        BOOLEAN NOT NULL DEFAULT FALSE,

    UNIQUE KEY uq_partnership (match_id, innings_no, wicket_no),
    KEY ix_partnership_match (match_id),
    CONSTRAINT fk_partnership_match FOREIGN KEY (match_id) REFERENCES matches (id) ON DELETE CASCADE,
    CONSTRAINT fk_partnership_striker FOREIGN KEY (striker_id) REFERENCES players (id),
    CONSTRAINT fk_partnership_non_striker FOREIGN KEY (non_striker_id) REFERENCES players (id),
    CONSTRAINT fk_partnership_team FOREIGN KEY (team_id) REFERENCES teams (id)
) ENGINE=InnoDB;

CREATE TABLE deliveries (
    id                   INT AUTO_INCREMENT PRIMARY KEY,
    match_id             INT NOT NULL,
    innings_no           INT NOT NULL,
    over_no              INT NOT NULL,
    ball_no              INT NOT NULL,
    ball_seq             INT NOT NULL,

    batting_team_id      INT,
    bowling_team_id      INT,
    batter_id            INT,
    non_striker_id       INT,
    bowler_id            INT,

    batter_runs          INT DEFAULT 0,
    extra_runs           INT DEFAULT 0,
    total_runs           INT DEFAULT 0,

    is_wide              BOOLEAN NOT NULL DEFAULT FALSE,
    is_no_ball           BOOLEAN NOT NULL DEFAULT FALSE,
    is_bye               BOOLEAN NOT NULL DEFAULT FALSE,
    is_leg_bye           BOOLEAN NOT NULL DEFAULT FALSE,
    is_legal             BOOLEAN NOT NULL DEFAULT TRUE,
    is_four              BOOLEAN NOT NULL DEFAULT FALSE,
    is_six               BOOLEAN NOT NULL DEFAULT FALSE,
    is_wicket            BOOLEAN NOT NULL DEFAULT FALSE,
    wicket_type          VARCHAR(40),
    dismissed_player_id  INT,

    cumulative_runs      INT,
    cumulative_wickets   INT,

    UNIQUE KEY uq_delivery (match_id, innings_no, ball_seq),
    KEY ix_deliveries_match         (match_id),
    KEY ix_deliveries_match_innings (match_id, innings_no),
    KEY ix_deliveries_batter        (batter_id),
    KEY ix_deliveries_bowler        (bowler_id),
    CONSTRAINT fk_deliveries_match  FOREIGN KEY (match_id)  REFERENCES matches (id) ON DELETE CASCADE,
    CONSTRAINT fk_deliveries_batter FOREIGN KEY (batter_id) REFERENCES players (id),
    CONSTRAINT fk_deliveries_bowler FOREIGN KEY (bowler_id) REFERENCES players (id)
) ENGINE=InnoDB;

CREATE TABLE match_players (
    id               INT AUTO_INCREMENT PRIMARY KEY,
    match_id         INT NOT NULL,
    team_id          INT NOT NULL,
    player_id        INT NOT NULL,

    is_playing_xi    BOOLEAN NOT NULL DEFAULT TRUE,
    is_captain       BOOLEAN NOT NULL DEFAULT FALSE,
    is_wicketkeeper  BOOLEAN NOT NULL DEFAULT FALSE,
    is_overseas      BOOLEAN,
    is_impact_sub    BOOLEAN NOT NULL DEFAULT FALSE,
    playing_order    INT,
    role             VARCHAR(40),

    UNIQUE KEY uq_match_player (match_id, team_id, player_id),
    KEY ix_match_players_match  (match_id),
    KEY ix_match_players_player (player_id),
    CONSTRAINT fk_mp_match  FOREIGN KEY (match_id)  REFERENCES matches (id) ON DELETE CASCADE,
    CONSTRAINT fk_mp_team   FOREIGN KEY (team_id)   REFERENCES teams (id),
    CONSTRAINT fk_mp_player FOREIGN KEY (player_id) REFERENCES players (id)
) ENGINE=InnoDB;

-- ===========================================================================
-- Operations
-- ===========================================================================
CREATE TABLE ingestion_runs (
    id                   INT AUTO_INCREMENT PRIMARY KEY,
    started_at           TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    finished_at          DATETIME,
    status               VARCHAR(20) NOT NULL DEFAULT 'running',
    `trigger`            VARCHAR(30),
    sources              VARCHAR(120),
    seasons              VARCHAR(200),

    matches_seen         INT DEFAULT 0,
    matches_inserted     INT DEFAULT 0,
    matches_updated      INT DEFAULT 0,
    matches_skipped      INT DEFAULT 0,
    deliveries_inserted  INT DEFAULT 0,
    duration_seconds     FLOAT,
    message              TEXT
) ENGINE=InnoDB;

-- ===========================================================================
-- Convenience view
-- ===========================================================================
CREATE OR REPLACE VIEW v_matches AS
SELECT
    m.id            AS match_id,
    m.match_key,
    m.season,
    m.match_date,
    m.stage,
    m.is_playoff,
    v.name          AS venue,
    m.city,
    t1.name         AS team1,
    t2.name         AS team2,
    tt.name         AS toss_winner,
    m.toss_decision,
    tf.name         AS first_batting_team,
    tw.name         AS winner,
    m.result_type,
    m.win_margin_runs,
    m.win_margin_wickets,
    m.result_summary,
    p.name          AS player_of_match,
    m.is_completed
FROM matches m
LEFT JOIN venues  v  ON v.id  = m.venue_id
LEFT JOIN teams   t1 ON t1.id = m.team1_id
LEFT JOIN teams   t2 ON t2.id = m.team2_id
LEFT JOIN teams   tt ON tt.id = m.toss_winner_id
LEFT JOIN teams   tf ON tf.id = m.first_batting_team_id
LEFT JOIN teams   tw ON tw.id = m.winner_id
LEFT JOIN players p  ON p.id  = m.player_of_match_id;
