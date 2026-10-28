let csrf = '';
let sessionRequest = null;

function responseError(data, status) {
 const error = new Error(data.error || `Request failed (${status})`);
 error.status = status;
 return error;
}

async function request(path, options = {}) {
 const method = (options.method || 'GET').toUpperCase();
 let response;
 try {
  response = await fetch(`/api${path}`, {
   credentials:'same-origin',
   ...options,
   headers:{
    ...(options.body && !(options.body instanceof FormData) ? {'Content-Type':'application/json'} : {}),
    ...(method !== 'GET' && method !== 'HEAD' && method !== 'OPTIONS' ? {'X-CSRF-Token':csrf} : {}),
    ...options.headers,
   },
  });
 } catch {
  throw new Error('Cannot reach the server. Start the backend and try again.');
 }
 const data = await response.json().catch(() => ({}));
 if (!response.ok) throw responseError(data, response.status);
 if (data.csrf_token) csrf = data.csrf_token;
 return data;
}

function loadSession() {
 if (!sessionRequest) {
  let pending;
  pending = request('/session').finally(() => {
   if (sessionRequest === pending) sessionRequest = null;
  });
  sessionRequest = pending;
 }
 return sessionRequest;
}

export async function api(path, options = {}) {
 const method = (options.method || 'GET').toUpperCase();
 if (path.split('?')[0] === '/session' && method === 'GET') return loadSession();
 if (method !== 'GET' && method !== 'HEAD' && method !== 'OPTIONS' && !csrf) await loadSession();
 return request(path, options);
}
