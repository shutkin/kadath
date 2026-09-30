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
    ask_delta     bytea,
    bid_delta     bytea
);
create index on order_book (created);
create index on order_book (instrument_id);

create table order_book_ask
(
    order_book_id bigint,
    price_id      integer,
    size          integer
);
create index on order_book_ask (order_book_id);

create table order_book_bid
(
    order_book_id bigint,
    price_id      integer,
    size          integer
);
create index on order_book_bid (order_book_id);

create table price
(
    id    serial,
    price numeric
);

create table spread_deal
(
    id serial,
    started timestamp with time zone not null,
    instrument_id integer not null,
    direction char not null,
    hold smallint not null,
    open_price_expected numeric,
    close_price_expected numeric,
    open_order_id varchar,
    close_order_id varchar
);
create index on spread_deal (started);

drop table deal;
create table deal
(
    id serial,
    started timestamp with time zone not null,
    leg1_instrument_id integer not null,
    leg2_instrument_id integer not null,
    direction char not null,
    hold smallint not null,
    leg1_status char not null,
    leg2_status char not null,
    leg1_open_price_expected numeric,
    leg2_open_price_expected numeric,
    leg1_close_price_expected numeric,
    leg2_close_price_expected numeric,
    leg1_open_order_id varchar,
    leg2_open_order_id varchar,
    leg1_close_order_id varchar,
    leg2_close_order_id varchar
);
create index on deal (started);

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