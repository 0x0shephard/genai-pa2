"""Read Kaggle quota/live logs with bounded waits and one OAuth refresh on HTTP 401.

Credentials stay in Kaggle's standard credential store; logs go to ignored outputs/.
"""
import argparse
import json
from pathlib import Path
import signal

from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk import KaggleCredentials
from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelSessionLogsStreamRequest
from requests.exceptions import HTTPError


class ObservationEnded(BaseException):
    """Escape stream reads without being wrapped as a network error."""


def timeout_handler(*_):
    raise ObservationEnded('Log observation window ended')


def observe(api, args):
    with api.build_kaggle_client() as client:
        service = client.kernels.kernels_api_client
        if args.action == 'quota':
            quota = service.get_accelerator_quota_statistics()
            print(json.dumps({
                'refresh': str(quota.quota_refresh_time),
                'gpu_used': str(quota.gpu_quota.time_used),
                'gpu_reserved': str(quota.gpu_quota.time_reserved),
                'gpu_allowed': str(quota.gpu_quota.total_time_allowed),
            }))
            return
        owner, slug = args.kernel.split('/')
        request = ApiGetKernelSessionLogsStreamRequest()
        request.user_name = owner
        request.kernel_slug = slug
        request.wait_for_logs_url_seconds = 5
        response = None
        destination = Path('outputs/kaggle-control') / slug / 'live.log'
        destination.parent.mkdir(parents=True, exist_ok=True)
        previous_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(args.seconds)
        try:
            response = service.get_kernel_session_logs_stream(request)
            with destination.open('w') as output:
                for line in response.iter_lines():
                    if line:
                        output.write(line.decode() + '\n')
                        output.flush()
        except ObservationEnded:
            pass
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous_handler)
            if response is not None:
                response.close()
        print(str(destination))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['quota', 'logs'])
    parser.add_argument('--kernel')
    parser.add_argument('--seconds', type=int, default=15)
    args = parser.parse_args()
    if args.action == 'logs' and not args.kernel:
        parser.error('--kernel is required for logs')
    if not 1 <= args.seconds <= 60:
        parser.error('--seconds must be between 1 and 60')
    api = KaggleApi()
    api.authenticate()
    try:
        observe(api, args)
    except HTTPError as error:
        if error.response.status_code != 401:
            raise
        with api.build_kaggle_client() as client:
            credentials = KaggleCredentials.load(client=client)
            if credentials is None:
                raise RuntimeError('Run kaggle auth login to reconnect your account') from error
            credentials.refresh_access_token()
        api = KaggleApi()
        api.authenticate()
        observe(api, args)


if __name__ == '__main__':
    main()
