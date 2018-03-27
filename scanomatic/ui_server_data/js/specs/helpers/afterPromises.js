/* This registers a callback to be run after promises have resolved.
 * It just calls setTimeout but has a more meaningful name.
 * It is based on the assumption that timeouts callbacks are handled after
 * promises, even if registered earlier.  This seems to be the consensual
 * interpretation of the Promise spec and it is the case it latest version of
 * most browsers.  (not older version of, for instance, Firefox though...).
 * It matches the chrome implementation, which is what is used to run the
 * tests.
 * */
export default function afterPromises(callback) {
    setTimeout(callback, 0);
}
