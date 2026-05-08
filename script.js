import {
  getDriverStandings,
  getConstructorStandings,
  getRaceSchedule
} from './api.js';

let driverContainer;
let constructorContainer;
let raceNameElement;
let raceDateElement;
let raceCircuitElement;
let countdownElement;
let countdownTimer;

function formatRaceDate(dateString, timeString) {
  const date = new Date(`${dateString}T${timeString || '12:00:00Z'}`);
  return date.toLocaleString('en-US', {
    weekday: 'long',
    month: 'long',
    day: 'numeric',
    year: 'numeric',
    hour: 'numeric',
    minute: '2-digit',
    timeZoneName: 'short'
  });
}

function getNextRace(races) {
  const now = new Date();
  return races
    .map(race => ({
      ...race,
      raceDate: new Date(`${race.date}T${race.time || '12:00:00Z'}`)
    }))
    .filter(race => race.raceDate > now)
    .sort((a, b) => a.raceDate - b.raceDate)[0];
}

function formatCountdown(diffMs) {
  if (diffMs <= 0) {
    return 'Race is starting now!';
  }

  const totalSeconds = Math.floor(diffMs / 1000);
  const days = Math.floor(totalSeconds / 86400);
  const hours = Math.floor((totalSeconds % 86400) / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const seconds = totalSeconds % 60;

  return `${days}d ${hours}h ${minutes}m ${seconds}s`;
}

async function loadDrivers() {
  try {
    const drivers = await getDriverStandings();

    driverContainer.innerHTML = '';

    drivers.slice(0, 5).forEach(driver => {
      const row = document.createElement('div');
      row.className = 'driver-row';

      row.innerHTML = `
        <div>
          <div class="driver-name">
            ${driver.Driver.givenName} ${driver.Driver.familyName}
          </div>
          <div class="team">
            ${driver.Constructors[0].name}
          </div>
        </div>
        <div class="points">
          ${driver.points}
        </div>
      `;

      driverContainer.appendChild(row);
    });
  } catch (error) {
    console.error('Error loading drivers:', error);
  }
}

async function loadConstructors() {
  try {
    const constructors = await getConstructorStandings();

    constructorContainer.innerHTML = '';

    constructors.slice(0, 5).forEach(team => {
      const row = document.createElement('div');
      row.className = 'driver-row';

      row.innerHTML = `
        <div>
          <div class="driver-name">
            ${team.Constructor.name}
          </div>
        </div>
        <div class="points">
          ${team.points}
        </div>
      `;

      constructorContainer.appendChild(row);
    });
  } catch (error) {
    console.error('Error loading constructors:', error);
  }
}

async function loadNextRace() {
  try {
    const races = await getRaceSchedule();
    const nextRace = getNextRace(races);

    if (!nextRace) {
      raceNameElement.textContent = 'Season complete';
      raceDateElement.textContent = '';
      raceCircuitElement.textContent = '';
      countdownElement.textContent = 'No upcoming race found.';
      return;
    }

    const targetDate = new Date(`${nextRace.date}T${nextRace.time || '12:00:00Z'}`);

    raceNameElement.textContent = `${nextRace.raceName}`;
    raceDateElement.textContent = formatRaceDate(nextRace.date, nextRace.time);
    raceCircuitElement.textContent = `${nextRace.Circuit.circuitName} · ${nextRace.Circuit.Location.locality}, ${nextRace.Circuit.Location.country}`;

    if (countdownTimer) {
      clearInterval(countdownTimer);
    }

    countdownElement.textContent = formatCountdown(targetDate - new Date());
    countdownTimer = setInterval(() => {
      countdownElement.textContent = formatCountdown(targetDate - new Date());
    }, 1000);
  } catch (error) {
    console.error('Error loading next race:', error);
    countdownElement.textContent = 'Unable to load next race.';
  }
}

function initDashboard() {
  driverContainer = document.getElementById('driver-standings');
  constructorContainer = document.getElementById('constructor-standings');
  raceNameElement = document.getElementById('race-name');
  raceDateElement = document.getElementById('race-date');
  raceCircuitElement = document.getElementById('race-circuit');
  countdownElement = document.getElementById('countdown');

  loadDrivers();
  loadConstructors();
  loadNextRace();
}

window.addEventListener('DOMContentLoaded', initDashboard);
