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
    open_order_id varchar,
    close_order_id varchar
);
create index on spread_deal (started);

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
