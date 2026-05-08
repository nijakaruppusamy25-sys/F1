const BASE_URL = "https://api.jolpi.ca/ergast/f1/current";

export async function getDriverStandings() {
  const response = await fetch(
    `${BASE_URL}/driverStandings.json`
  );

  const data = await response.json();

  return data.MRData.StandingsTable.StandingsLists[0].DriverStandings;
}

export async function getConstructorStandings() {
  const response = await fetch(
    `${BASE_URL}/constructorStandings.json`
  );

  const data = await response.json();

  return data.MRData.StandingsTable.StandingsLists[0].ConstructorStandings;
}

export async function getRaceSchedule() {
  const response = await fetch(
    `${BASE_URL}/races.json`
  );

  const data = await response.json();

  return data.MRData.RaceTable.Races;
}