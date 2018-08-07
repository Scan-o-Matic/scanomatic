import $ from 'jquery';

export default class API {
    static handleSuccess(callbackSuccess, callbackFail) {
        return (data) => {
            let response;
            try {
                response = JSON.parse(data);
            } catch (err) {
                return callbackSuccess(data);
            }
            if (response && response.success === false) return callbackFail(data.reason);
            return callbackSuccess(data);
        };
    }

    static handleFail(callbackFail) {
        return (jqXHR) => {
            let response;
            try {
                response = JSON.parse(jqXHR.responseText);
            } catch (err) {
                return callbackFail('Unexpected server error');
            }
            return callbackFail(response.reason);
        };
    }

    static get(url) {
        return new Promise((resolve, reject) => $.ajax({
            url,
            type: 'GET',
            success: API.handleSuccess(resolve, reject),
            error: API.handleFail(reject),
        }));
    }

    static delete(url) {
        return new Promise((resolve, reject) => $.ajax({
            url,
            type: 'DELETE',
            success: API.handleSuccess(resolve, reject),
            error: API.handleFail(reject),
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
            success: API.handleSuccess(resolve, reject),
            error: API.handleFail(reject),
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
                API.handleSuccess(resolve, reject),
                API.handleFail(reject),
            ));
    }
}
