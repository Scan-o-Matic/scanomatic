import $ from 'jquery';

export default class API {
    static get(url) {
        return new Promise((resolve, reject) => $.ajax({
            url,
            type: 'GET',
            success: resolve,
            error: jqXHR => reject(JSON.parse(jqXHR.responseText).reason),
        }));
    }

    static delete(url) {
        return new Promise((resolve, reject) => $.ajax({
            url,
            type: 'DELETE',
            success: resolve,
            error: jqXHR => reject(JSON.parse(jqXHR.responseText).reason),
        }));
    }

    static postFormData(url, formData) {
        return new Promise((resolve, reject) => $.ajax({
            url,
            type: 'POST',
            contentType: false,
            enctype: 'multipart/form-data',
            data: formData,
            processData: false,
            success: resolve,
            error: jqXHR => reject(JSON.parse(jqXHR.responseText).reason),
        }));
    }

    static postJSON(url, json) {
        return new Promise((resolve, reject) => $.ajax({
            url,
            type: 'POST',
            data: JSON.stringify(json),
            contentType: 'application/json',
        })
            .then(
                resolve,
                jqXHR => reject(JSON.parse(jqXHR.responseText).reason),
            ));
    }
}
