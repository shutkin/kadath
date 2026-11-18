create table instrument
(
    id serial primary key,
    ticker varchar,
    name varchar,
    lot_size smallint
);
create index on instrument (ticker);

create table order_book
(
    id            bigserial,
    created       timestamp with time zone default now() not null,
    instrument_id integer not null,
    ask     bytea,
    bid     bytea
);
create index on order_book (created);
create index on order_book (instrument_id);

create table price
(
    id    serial,
    price numeric
);

create table deal
(
    id serial,
    instrument_id integer not null,
    lots smallint not null,
    direction char not null,
    max_hold integer not null,

    open_time timestamp with time zone not null,
    expected_open_price numeric not null,
    open_order_id varchar,
    open_status char not null,

    close_time timestamp with time zone,
    expected_close_price numeric,
    close_order_id varchar,
    close_status char not null
);
create index on deal (instrument_id);
create index on deal (open_time);

create table trade
(
    id serial,
    order_id varchar not null,
    executed timestamp with time zone not null,
    direction char not null,
    price numeric not null,
    quantity smallint not null,
    json varchar
);
create index on trade (order_id);
create index on trade (executed);

create table signal_calculator
(
    id serial primary key,
    trained_on timestamp with time zone not null,
    up_score float4,
    up_params text,
    down_score float4,
    down_params text
);
create index on signal_calculator (trained_on);

alter table instrument add column source char default 'A' not null;
comment on column instrument.source is 'A - Alor, T - T-Invest';

alter table instrument add column external_id varchar;

alter table signal_calculator add column train_time_ms int;

create table test_session
(
    id serial primary key,
    started timestamp with time zone not null,
    instrument_id integer not null
);

create table test_signals
(
    id bigserial primary key,
    session_id integer not null,
    event_time timestamp with time zone not null,
    signal1 float4,
    signal2 float4,
    signal3 float4
);
create index on test_signals (session_id);
create index on test_signals (event_time);

create table test_deal
(
    id serial primary key,
    session_id integer not null,
    direction char not null,
    open_time timestamp with time zone not null,
    close_time timestamp with time zone,
    open_price float4,
    close_price float4
);
create index on test_deal (session_id);