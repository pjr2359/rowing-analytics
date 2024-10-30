--
-- PostgreSQL database dump
--

-- Dumped from database version 17.0
-- Dumped by pg_dump version 17.0

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET transaction_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: boat; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.boat (
    boat_id integer NOT NULL,
    name character varying(100) NOT NULL,
    boat_class character varying(10)
);


ALTER TABLE public.boat OWNER TO postgres;

--
-- Name: boat_boat_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.boat_boat_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.boat_boat_id_seq OWNER TO postgres;

--
-- Name: boat_boat_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.boat_boat_id_seq OWNED BY public.boat.boat_id;


--
-- Name: erg_data; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.erg_data (
    erg_data_id integer NOT NULL,
    rower_id integer,
    test_date date,
    overall_split double precision,
    watts_per_lb double precision,
    weight double precision,
    pacing double precision[]
);


ALTER TABLE public.erg_data OWNER TO postgres;

--
-- Name: erg_data_erg_data_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.erg_data_erg_data_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.erg_data_erg_data_id_seq OWNER TO postgres;

--
-- Name: erg_data_erg_data_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.erg_data_erg_data_id_seq OWNED BY public.erg_data.erg_data_id;


--
-- Name: event; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.event (
    event_id integer NOT NULL,
    event_date date,
    event_name character varying(100)
);


ALTER TABLE public.event OWNER TO postgres;

--
-- Name: event_event_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.event_event_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.event_event_id_seq OWNER TO postgres;

--
-- Name: event_event_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.event_event_id_seq OWNED BY public.event.event_id;


--
-- Name: lineup; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.lineup (
    lineup_id integer NOT NULL,
    piece_id integer,
    boat_id integer,
    rower_id integer,
    seat_number integer,
    is_coxswain boolean DEFAULT false
);


ALTER TABLE public.lineup OWNER TO postgres;

--
-- Name: lineup_lineup_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.lineup_lineup_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.lineup_lineup_id_seq OWNER TO postgres;

--
-- Name: lineup_lineup_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.lineup_lineup_id_seq OWNED BY public.lineup.lineup_id;


--
-- Name: piece; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.piece (
    piece_id integer NOT NULL,
    event_id integer,
    piece_number integer,
    distance integer,
    description character varying(255)
);


ALTER TABLE public.piece OWNER TO postgres;

--
-- Name: piece_piece_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.piece_piece_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.piece_piece_id_seq OWNER TO postgres;

--
-- Name: piece_piece_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.piece_piece_id_seq OWNED BY public.piece.piece_id;


--
-- Name: result; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.result (
    result_id integer NOT NULL,
    piece_id integer,
    boat_id integer,
    "time" double precision,
    split double precision,
    margin double precision
);


ALTER TABLE public.result OWNER TO postgres;

--
-- Name: result_result_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.result_result_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.result_result_id_seq OWNER TO postgres;

--
-- Name: result_result_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.result_result_id_seq OWNED BY public.result.result_id;


--
-- Name: rower; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.rower (
    rower_id integer NOT NULL,
    name character varying(100) NOT NULL,
    weight double precision,
    side character varying(10)
);


ALTER TABLE public.rower OWNER TO postgres;

--
-- Name: rower_rower_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.rower_rower_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.rower_rower_id_seq OWNER TO postgres;

--
-- Name: rower_rower_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.rower_rower_id_seq OWNED BY public.rower.rower_id;


--
-- Name: seat_race; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.seat_race (
    seat_race_id integer NOT NULL,
    event_id integer,
    piece_numbers integer[],
    rower_id_1 integer,
    rower_id_2 integer,
    time_difference double precision,
    winner_id integer,
    notes text
);


ALTER TABLE public.seat_race OWNER TO postgres;

--
-- Name: seat_race_seat_race_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.seat_race_seat_race_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.seat_race_seat_race_id_seq OWNER TO postgres;

--
-- Name: seat_race_seat_race_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.seat_race_seat_race_id_seq OWNED BY public.seat_race.seat_race_id;


--
-- Name: boat boat_id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.boat ALTER COLUMN boat_id SET DEFAULT nextval('public.boat_boat_id_seq'::regclass);


--
-- Name: erg_data erg_data_id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.erg_data ALTER COLUMN erg_data_id SET DEFAULT nextval('public.erg_data_erg_data_id_seq'::regclass);


--
-- Name: event event_id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.event ALTER COLUMN event_id SET DEFAULT nextval('public.event_event_id_seq'::regclass);


--
-- Name: lineup lineup_id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.lineup ALTER COLUMN lineup_id SET DEFAULT nextval('public.lineup_lineup_id_seq'::regclass);


--
-- Name: piece piece_id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.piece ALTER COLUMN piece_id SET DEFAULT nextval('public.piece_piece_id_seq'::regclass);


--
-- Name: result result_id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.result ALTER COLUMN result_id SET DEFAULT nextval('public.result_result_id_seq'::regclass);


--
-- Name: rower rower_id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.rower ALTER COLUMN rower_id SET DEFAULT nextval('public.rower_rower_id_seq'::regclass);


--
-- Name: seat_race seat_race_id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.seat_race ALTER COLUMN seat_race_id SET DEFAULT nextval('public.seat_race_seat_race_id_seq'::regclass);


--
-- Data for Name: boat; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.boat (boat_id, name, boat_class) FROM stdin;
1	Schlaepfer II	Unknown
2	Defiance	Unknown
3	Class of 1977	Unknown
4	Sledgehammer	Unknown
5	Class of 77	Unknown
6	Dupcak	Unknown
7	3/4	Unknown
8	Swinney over Bechard by 0.65 secs	Unknown
9	White Stripes	Unknown
10	Harney	Unknown
11	Shed	Unknown
12	Jay Abbe	Unknown
13	Karen Abbe	Unknown
14	Steinl	Unknown
15	Dundon over Forg by a total 16.95 secs	Unknown
16	Class of '77	Unknown
\.


--
-- Data for Name: erg_data; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.erg_data (erg_data_id, rower_id, test_date, overall_split, watts_per_lb, weight, pacing) FROM stdin;
\.


--
-- Data for Name: event; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.event (event_id, event_date, event_name) FROM stdin;
1	2024-10-03	Race on 2024-10-03
2	2024-09-14	Race on 2024-09-14
3	2024-09-25	Race on 2024-09-25
5	2024-10-03	Race on 2024-10-03
6	2024-09-14	Race on 2024-09-14
7	2024-09-25	Race on 2024-09-25
8	\N	Schwartz Cup
\.


--
-- Data for Name: lineup; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.lineup (lineup_id, piece_id, boat_id, rower_id, seat_number, is_coxswain) FROM stdin;
1	1	1	1	0	t
2	1	1	5	8	f
3	1	1	8	7	f
4	1	1	11	6	f
5	1	1	13	5	f
6	1	1	16	4	f
7	1	1	20	3	f
8	1	1	24	2	f
9	1	1	28	1	f
10	1	2	2	0	t
11	1	2	6	8	f
12	1	2	9	7	f
13	1	2	8	6	f
14	1	2	14	5	f
15	1	2	17	4	f
16	1	2	21	3	f
17	1	2	25	2	f
18	1	2	29	1	f
19	1	3	3	0	t
20	1	3	7	8	f
21	1	3	10	7	f
22	1	3	12	6	f
23	1	3	15	5	f
24	1	3	18	4	f
25	1	3	22	3	f
26	1	3	26	2	f
27	1	3	30	1	f
28	1	4	4	0	t
29	1	4	19	4	f
30	1	4	23	3	f
31	1	4	27	2	f
32	1	4	31	1	f
33	5	2	1	0	t
34	5	2	20	8	f
35	5	2	11	7	f
36	5	2	30	6	f
37	5	2	5	5	f
38	5	2	6	4	f
39	5	2	33	3	f
40	5	2	36	2	f
41	5	2	37	1	f
42	5	5	3	0	t
43	5	5	24	8	f
44	5	5	8	7	f
45	5	5	21	6	f
46	5	5	13	5	f
47	5	5	35	4	f
48	5	5	25	3	f
49	5	5	32	2	f
50	5	5	17	1	f
51	5	6	2	0	t
52	5	6	29	8	f
53	5	6	19	7	f
54	5	6	18	6	f
55	5	6	34	5	f
56	5	6	7	4	f
57	5	6	15	3	f
58	5	6	26	2	f
59	5	6	38	1	f
60	8	9	8	4	f
61	8	9	11	3	f
62	8	9	20	2	f
63	8	9	24	1	f
64	8	10	28	4	f
65	8	10	35	3	f
66	8	10	6	2	f
67	8	10	5	1	f
68	8	11	32	4	f
69	8	11	13	3	f
70	8	11	33	2	f
71	8	11	36	1	f
72	8	12	1	0	t
73	8	12	8	4	f
74	8	12	14	3	f
75	8	12	17	2	f
76	8	12	26	1	f
77	8	4	3	0	t
78	8	4	10	4	f
79	8	4	7	3	f
80	8	4	15	2	f
81	8	4	18	1	f
82	8	13	39	0	t
83	8	13	37	4	f
84	8	13	25	3	f
85	8	13	12	2	f
86	8	13	34	1	f
87	8	14	2	0	t
88	8	14	31	4	f
89	8	14	38	3	f
90	8	14	23	2	f
91	8	14	22	1	f
92	12	1	1	0	t
93	12	1	5	8	f
94	12	1	8	7	f
95	12	1	11	6	f
96	12	1	13	5	f
97	12	1	16	4	f
98	12	1	20	3	f
99	12	1	24	2	f
100	12	1	28	1	f
101	12	2	2	0	t
102	12	2	6	8	f
103	12	2	9	7	f
104	12	2	8	6	f
105	12	2	14	5	f
106	12	2	17	4	f
107	12	2	21	3	f
108	12	2	25	2	f
109	12	2	29	1	f
110	12	3	3	0	t
111	12	3	7	8	f
112	12	3	10	7	f
113	12	3	12	6	f
114	12	3	15	5	f
115	12	3	18	4	f
116	12	3	22	3	f
117	12	3	26	2	f
118	12	3	30	1	f
119	12	4	4	0	t
120	12	4	19	4	f
121	12	4	23	3	f
122	12	4	27	2	f
123	12	4	31	1	f
124	16	2	1	0	t
125	16	2	20	8	f
126	16	2	11	7	f
127	16	2	30	6	f
128	16	2	5	5	f
129	16	2	6	4	f
130	16	2	33	3	f
131	16	2	36	2	f
132	16	2	37	1	f
133	16	5	3	0	t
134	16	5	24	8	f
135	16	5	8	7	f
136	16	5	21	6	f
137	16	5	13	5	f
138	16	5	35	4	f
139	16	5	25	3	f
140	16	5	32	2	f
141	16	5	17	1	f
142	16	6	2	0	t
143	16	6	29	8	f
144	16	6	19	7	f
145	16	6	18	6	f
146	16	6	34	5	f
147	16	6	7	4	f
148	16	6	15	3	f
149	16	6	26	2	f
150	16	6	38	1	f
151	19	9	8	4	f
152	19	9	11	3	f
153	19	9	20	2	f
154	19	9	24	1	f
155	19	10	28	4	f
156	19	10	35	3	f
157	19	10	6	2	f
158	19	10	5	1	f
159	19	11	32	4	f
160	19	11	13	3	f
161	19	11	33	2	f
162	19	11	36	1	f
163	19	12	1	0	t
164	19	12	8	4	f
165	19	12	14	3	f
166	19	12	17	2	f
167	19	12	26	1	f
168	19	4	3	0	t
169	19	4	10	4	f
170	19	4	7	3	f
171	19	4	15	2	f
172	19	4	18	1	f
173	19	13	39	0	t
174	19	13	37	4	f
175	19	13	25	3	f
176	19	13	12	2	f
177	19	13	34	1	f
178	19	14	2	0	t
179	19	14	31	4	f
180	19	14	38	3	f
181	19	14	23	2	f
182	19	14	22	1	f
183	23	16	3	0	t
184	23	16	11	8	f
185	23	16	25	7	f
186	23	16	37	6	f
187	23	16	8	5	f
188	23	16	18	4	f
189	23	16	19	3	f
190	23	16	24	2	f
191	23	16	28	1	f
192	23	2	1	0	t
193	23	2	20	8	f
194	23	2	5	7	f
195	23	2	10	6	f
196	23	2	7	5	f
197	23	2	27	4	f
198	23	2	40	3	f
199	23	2	32	2	f
200	23	2	21	1	f
201	23	1	39	0	t
202	23	1	14	8	f
203	23	1	8	7	f
204	23	1	12	6	f
205	23	1	38	5	f
206	23	1	23	4	f
207	23	1	15	3	f
208	23	1	31	2	f
209	23	1	36	1	f
210	23	6	4	0	t
211	23	6	6	8	f
212	23	6	29	7	f
213	23	6	13	6	f
214	23	6	35	5	f
215	23	6	22	4	f
216	23	6	33	3	f
217	23	6	17	2	f
218	23	6	26	1	f
\.


--
-- Data for Name: piece; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.piece (piece_id, event_id, piece_number, distance, description) FROM stdin;
1	1	1	\N	Piece 1
2	1	2	\N	Piece 2
3	1	3	\N	Piece 3
4	1	4	\N	Piece 4
5	2	1	\N	Piece 1
6	2	2	\N	Piece 2
7	2	3	\N	Piece 3
8	3	1	\N	Piece 1
9	3	2	\N	Piece 2
10	3	3	\N	Piece 3
11	3	4	\N	Piece 4
12	5	1	\N	Piece 1
13	5	2	\N	Piece 2
14	5	3	\N	Piece 3
15	5	4	\N	Piece 4
16	6	1	\N	Piece 1
17	6	2	\N	Piece 2
18	6	3	\N	Piece 3
19	7	1	\N	Piece 1
20	7	2	\N	Piece 2
21	7	3	\N	Piece 3
22	7	4	\N	Piece 4
23	8	1	\N	Piece 1
24	8	2	\N	Piece 2
\.


--
-- Data for Name: result; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.result (result_id, piece_id, boat_id, "time", split, margin) FROM stdin;
1	1	1	376.3	94.08	\N
44	10	4	473.65999999999997	118.42	4.76
45	10	13	465.90999999999997	116.47999999999999	7.75
46	10	14	494.99	123.75	29.08
2	1	2	389.9	97.47999999999999	13.6
3	1	3	388.9	97.22999999999999	1
4	1	4	478.8	119.7	89.9
5	2	1	402.1	100.53	\N
6	2	2	409.6	102.4	7.5
7	2	3	418.8	104.7	9.2
8	2	4	522.1	130.53	103.3
9	3	1	378.4	94.6	\N
10	3	2	392.6	98.15	14.2
11	3	3	387.9	96.97999999999999	4.7
12	3	4	472.8	118.2	84.9
13	4	1	397.4	99.35	\N
47	11	9	402.5	100.63	\N
70	16	2	751.05	96.09	\N
14	4	2	415.6	103.9	18.2
15	4	3	406.6	101.65	9
16	4	4	514.2	128.55	107.6
17	5	2	751.05	96.09	\N
85	19	14	493.01	123.25	18.47
18	5	5	770.99	98.27000000000001	19.94
19	5	6	811.77	103.52000000000001	40.78
20	6	2	752.03	99.66	\N
71	16	5	770.99	98.27000000000001	19.94
72	16	6	811.77	103.52000000000001	40.78
21	6	5	775.61	102.24000000000001	23.58
22	6	6	811.4	106.6	35.79
23	7	2	749.18	96.02000000000001	\N
48	11	10	406.28	101.57	3.78
24	7	5	768.47	98.52000000000001	19.29
25	7	6	813.96	104.09	45.49
26	8	9	435.92	108.97999999999999	\N
49	11	11	412.55	103.14	6.27
50	11	12	431.74	107.94	19.19
51	11	4	428.72	107.18	3.02
52	11	13	438.19	109.55	9.47
53	11	14	447.71	111.93	9.52
27	8	10	439.13	109.78	3.21
28	8	11	452.5	113.13	13.37
29	8	12	472.59000000000003	118.15	20.09
30	8	4	487.76	121.94	15.17
31	8	13	474.54	118.64	13.22
32	8	14	493.01	123.25	18.47
33	9	9	404.12	101.03	\N
54	12	1	376.3	94.08	\N
34	9	10	403.11	100.78	1.01
35	9	11	412.68	103.17	9.57
36	9	12	447.74	111.94	35.06
37	9	4	437.95	109.49000000000001	9.79
38	9	13	435.94	108.99000000000001	2.01
39	9	14	458.31	114.58	22.37
40	10	9	435.27	108.82	\N
73	17	2	752.03	99.66	\N
55	12	2	389.9	97.47999999999999	13.6
56	12	3	388.9	97.22999999999999	1
74	17	5	775.61	102.24000000000001	23.58
41	10	10	439.54	109.89	4.27
42	10	11	444.93	111.22999999999999	5.39
43	10	12	468.9	117.22999999999999	23.97
57	12	4	478.8	119.7	89.9
58	13	1	402.1	100.53	\N
75	17	6	811.4	106.6	35.79
59	13	2	409.6	102.4	7.5
60	13	3	418.8	104.7	9.2
61	13	4	522.1	130.53	103.3
62	14	1	378.4	94.6	\N
76	18	2	749.18	96.02000000000001	\N
63	14	2	392.6	98.15	14.2
64	14	3	387.9	96.97999999999999	4.7
65	14	4	472.8	118.2	84.9
66	15	1	397.4	99.35	\N
77	18	5	768.47	98.52000000000001	19.29
78	18	6	813.96	104.09	45.49
67	15	2	415.6	103.9	18.2
68	15	3	406.6	101.65	9
69	15	4	514.2	128.55	107.6
79	19	9	435.92	108.97999999999999	\N
86	20	9	404.12	101.03	\N
93	21	9	435.27	108.82	\N
96	21	12	468.9	117.22999999999999	23.97
97	21	4	473.65999999999997	118.42	4.76
80	19	10	439.13	109.78	3.21
81	19	11	452.5	113.13	13.37
82	19	12	472.59000000000003	118.15	20.09
83	19	4	487.76	121.94	15.17
84	19	13	474.54	118.64	13.22
100	22	9	402.5	100.63	\N
102	22	11	412.55	103.14	6.27
103	22	12	431.74	107.94	19.19
104	22	4	428.72	107.18	3.02
87	20	10	403.11	100.78	1.01
88	20	11	412.68	103.17	9.57
89	20	12	447.74	111.94	35.06
90	20	4	437.95	109.49000000000001	9.79
91	20	13	435.94	108.99000000000001	2.01
92	20	14	458.31	114.58	22.37
94	21	10	439.54	109.89	4.27
95	21	11	444.93	111.22999999999999	5.39
98	21	13	465.90999999999997	116.47999999999999	7.75
99	21	14	494.99	123.75	29.08
105	22	13	438.19	109.55	9.47
106	22	14	447.71	111.93	9.52
107	23	16	872.3	36.9	\N
101	22	10	406.28	101.57	3.78
108	23	2	876.9	37.4	\N
109	23	1	889.7	38.9	\N
110	23	6	889.1	38.8	\N
111	24	16	892.6	39.2	\N
112	24	2	915.4	41.7	\N
113	24	1	924.9	42.8	\N
114	24	6	913.8	41.5	\N
\.


--
-- Data for Name: rower; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.rower (rower_id, name, weight, side) FROM stdin;
1	Walsey	\N	\N
2	Lago	\N	\N
3	Vu	\N	\N
4	Goldstein	\N	\N
5	Swinney	\N	\N
6	Brown	\N	\N
7	Busby	\N	\N
8	Savell	\N	\N
9	Reilly/Purcea	\N	\N
10	El Hadj	\N	\N
11	Patterson	\N	\N
12	Dundon	\N	\N
13	Xu	\N	\N
14	Zaslow	\N	\N
15	Mayer	\N	\N
16	Purcea/Reilly	\N	\N
17	Bailey	\N	\N
18	Forg	\N	\N
19	Calalang	\N	\N
20	Hohlt	\N	\N
21	Genden	\N	\N
22	Zegger	\N	\N
23	Price	\N	\N
24	Albrecht	\N	\N
25	Oliveira	\N	\N
26	Holtman	\N	\N
27	Lau	\N	\N
28	Alston	\N	\N
29	Yang	\N	\N
30	Smith	\N	\N
31	Lujan	\N	\N
32	Reilly	\N	\N
33	Purcea	\N	\N
34	SoucieGarza	\N	\N
35	Bechard	\N	\N
36	Foxley	\N	\N
37	Lynch	\N	\N
38	Aghazadeh	\N	\N
39	Johnson	\N	\N
40	Phelps	\N	\N
\.


--
-- Data for Name: seat_race; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.seat_race (seat_race_id, event_id, piece_numbers, rower_id_1, rower_id_2, time_difference, winner_id, notes) FROM stdin;
1	1	{}	32	33	11.3	32	Reilly over Purcea by a total 11.3 secs
2	5	{}	32	33	11.3	32	Reilly over Purcea by a total 11.3 secs
\.


--
-- Name: boat_boat_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.boat_boat_id_seq', 16, true);


--
-- Name: erg_data_erg_data_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.erg_data_erg_data_id_seq', 1, false);


--
-- Name: event_event_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.event_event_id_seq', 8, true);


--
-- Name: lineup_lineup_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.lineup_lineup_id_seq', 218, true);


--
-- Name: piece_piece_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.piece_piece_id_seq', 24, true);


--
-- Name: result_result_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.result_result_id_seq', 114, true);


--
-- Name: rower_rower_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.rower_rower_id_seq', 40, true);


--
-- Name: seat_race_seat_race_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.seat_race_seat_race_id_seq', 2, true);


--
-- Name: boat boat_name_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.boat
    ADD CONSTRAINT boat_name_key UNIQUE (name);


--
-- Name: boat boat_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.boat
    ADD CONSTRAINT boat_pkey PRIMARY KEY (boat_id);


--
-- Name: erg_data erg_data_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.erg_data
    ADD CONSTRAINT erg_data_pkey PRIMARY KEY (erg_data_id);


--
-- Name: event event_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.event
    ADD CONSTRAINT event_pkey PRIMARY KEY (event_id);


--
-- Name: lineup lineup_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.lineup
    ADD CONSTRAINT lineup_pkey PRIMARY KEY (lineup_id);


--
-- Name: piece piece_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.piece
    ADD CONSTRAINT piece_pkey PRIMARY KEY (piece_id);


--
-- Name: result result_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.result
    ADD CONSTRAINT result_pkey PRIMARY KEY (result_id);


--
-- Name: rower rower_name_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.rower
    ADD CONSTRAINT rower_name_key UNIQUE (name);


--
-- Name: rower rower_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.rower
    ADD CONSTRAINT rower_pkey PRIMARY KEY (rower_id);


--
-- Name: seat_race seat_race_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.seat_race
    ADD CONSTRAINT seat_race_pkey PRIMARY KEY (seat_race_id);


--
-- Name: erg_data erg_data_rower_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.erg_data
    ADD CONSTRAINT erg_data_rower_id_fkey FOREIGN KEY (rower_id) REFERENCES public.rower(rower_id);


--
-- Name: lineup lineup_boat_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.lineup
    ADD CONSTRAINT lineup_boat_id_fkey FOREIGN KEY (boat_id) REFERENCES public.boat(boat_id);


--
-- Name: lineup lineup_piece_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.lineup
    ADD CONSTRAINT lineup_piece_id_fkey FOREIGN KEY (piece_id) REFERENCES public.piece(piece_id);


--
-- Name: lineup lineup_rower_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.lineup
    ADD CONSTRAINT lineup_rower_id_fkey FOREIGN KEY (rower_id) REFERENCES public.rower(rower_id);


--
-- Name: piece piece_event_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.piece
    ADD CONSTRAINT piece_event_id_fkey FOREIGN KEY (event_id) REFERENCES public.event(event_id);


--
-- Name: result result_boat_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.result
    ADD CONSTRAINT result_boat_id_fkey FOREIGN KEY (boat_id) REFERENCES public.boat(boat_id);


--
-- Name: result result_piece_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.result
    ADD CONSTRAINT result_piece_id_fkey FOREIGN KEY (piece_id) REFERENCES public.piece(piece_id);


--
-- Name: seat_race seat_race_event_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.seat_race
    ADD CONSTRAINT seat_race_event_id_fkey FOREIGN KEY (event_id) REFERENCES public.event(event_id);


--
-- Name: seat_race seat_race_rower_id_1_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.seat_race
    ADD CONSTRAINT seat_race_rower_id_1_fkey FOREIGN KEY (rower_id_1) REFERENCES public.rower(rower_id);


--
-- Name: seat_race seat_race_rower_id_2_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.seat_race
    ADD CONSTRAINT seat_race_rower_id_2_fkey FOREIGN KEY (rower_id_2) REFERENCES public.rower(rower_id);


--
-- Name: seat_race seat_race_winner_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.seat_race
    ADD CONSTRAINT seat_race_winner_id_fkey FOREIGN KEY (winner_id) REFERENCES public.rower(rower_id);


--
-- PostgreSQL database dump complete
--

