const secondsInADay = 86400;
const secondsInAnHour = 3600;
const secondsInAMinute = 60;

class Duration {
    constructor(seconds) {
        this.totalSeconds = seconds;
        Object.freeze(this);
    }

    get totalMilliseconds() {
        return this.totalSeconds * 1000;
    }

    get days() {
        return Math.floor(this.totalSeconds / secondsInADay);
    }

    get hours() {
        return Math.floor((this.totalSeconds % secondsInADay) / secondsInAnHour);
    }

    get minutes() {
        return Math.floor((this.totalSeconds % secondsInAnHour) / secondsInAMinute);
    }

    get seconds() {
        return this.totalSeconds % secondsInAMinute;
    }

    after(someDate) {
        return new Date(someDate.getTime() + this.totalMilliseconds);
    }

    shifted(days, hours, minutes) {
        return new Duration(((this.days + days) * secondsInADay) +
            ((this.hours + hours) * secondsInAnHour) +
            ((this.minutes + minutes) * secondsInAMinute));
    }

    before(someDate) {
        return new Date(someDate.getTime() - this.totalMilliseconds);
    }

    static fromMilliseconds(milliSeconds) {
        return new Duration(milliSeconds / 1000);
    }
}

export default Duration;
