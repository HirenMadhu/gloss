# Formula 1 results database

This is a historical Formula 1 motor-racing database derived from the Ergast schema. It records the
world championship from its first season onward: the calendar of races each season, the drivers and
constructors (teams) that entered, and the per-event outcomes — qualifying, race classifications, and
the running championship standings after each round. Most tables are timestamped by the date of the
race weekend they belong to; the driver, constructor, and circuit tables are slow-changing reference
("dimension") tables that describe entities rather than events.

A note on time: event tables carry a `date` column that is the date of the associated race. The
reference tables are largely static — a driver's row, for instance, is created once and rarely changes.

## races

One row per Grand Prix. `raceId` is the primary key. Each race happens at a venue, so `circuitId`
references the `circuits` table — this is the only foreign key on this table. `year` and `round` locate
the race within a season (round 1 is the season opener); together they're effectively a natural key.
`name` is the event's official name (e.g. "Monaco Grand Prix"). `date` is the race day and is what we
treat as the event time for everything that hangs off the race. There is also a `time` column holding
the scheduled start time of day; it is frequently missing for older seasons, so don't rely on it.

## circuits

The venues. `circuitId` is the primary key. A circuit has a human-readable `name`, a `location` (city)
and `country`, and geographic coordinates in `lat`/`lng` (decimal degrees). `alt` is the altitude of
the circuit in metres above sea level. `circuitRef` is a short stable slug used in URLs. This table has
no foreign keys — it sits at the "top" of the hierarchy and is referenced by `races`.

## drivers

The competitors. `driverId` is the primary key. `forename` and `surname` are the driver's name; `code`
is the three-letter abbreviation shown on TV timing screens (e.g. "HAM"), which is null for many older
drivers who predate the convention. `dob` is the date of birth. `nationality` is given as a demonym
("British", "German"). `driverRef` is a stable lower-case slug. No foreign keys — drivers are referenced
from the event tables.

## constructors

The teams that build and enter the cars. `constructorId` is the primary key; `name` is the team name
(e.g. "Ferrari", "McLaren") and `nationality` is the team's licensing nationality. Note that a driver
and a constructor are different things: a driver is a person, a constructor is the chassis/team. A single
race entry pairs one driver with one constructor.

## results

The race classification — one row per car per race, and the central fact table of the database.
`resultId` is the primary key. It carries three foreign keys, which is what makes a result meaningful:
`raceId` says which race, `driverId` who was driving, and `constructorId` which team's car they drove.
The two entity references (driver and constructor) are distinct roles and should not be conflated — the
same race has many results, each tying a different driver–constructor pairing to the event.

`grid` is the starting-grid position (1 = pole; 0 is used for a pit-lane start). `positionOrder` is the
finishing order used for sorting, always present. `points` is the championship points awarded for that
result under the scoring system in force that season. `laps` is the number of laps the car completed.
`statusId` encodes how the car's race ended — it references a status reference vocabulary whose codes
distinguish a normal classified finish from the many ways a car can fail to make the end (for example
accident, collision, or a mechanical retirement such as engine, gearbox, or hydraulics). A large share of
historical entries did not see the chequered flag, so this column matters. `milliseconds` is the total
race time in milliseconds for cars that were classified, and is null otherwise. `fastestLap` is the lap
number on which the driver set their fastest lap.

## qualifying

The Saturday qualifying session that sets the grid. `qualifyId` is the primary key. Like `results` it
references the race (`raceId`), the `driverId`, and the `constructorId` — again, driver and constructor
are separate roles. `position` is the qualifying classification. `number` is the car number.

## standings

The drivers' championship table as it stood after each race — i.e. a running cumulative snapshot, one
row per driver per race. `driverStandingsId` is the primary key; `raceId` and `driverId` say which race
and driver the snapshot is for. `points` is the season-to-date points total, `position` the current rank
in the championship, and `wins` the number of wins so far that season. Because it is cumulative, values
grow monotonically across the rounds of a season and reset between seasons.

## constructor_standings

The same idea as `standings`, but for the constructors' championship: a per-race cumulative snapshot of
each team's season-to-date `points`, championship `position`, and `wins`. `raceId` and `constructorId`
are the foreign keys.

<!-- constructor_results is intentionally left undocumented here; coverage is partial by design. -->
