"""
Pipeline EC2 Control Lambda
파이프라인 서버 EC2 인스턴스 시작/종료 제어
"""

import boto3
import time

ec2 = boto3.client('ec2', region_name='ap-northeast-2')

PIPELINE_INSTANCE_ID = 'i-079e9af2307dd34a1'


def lambda_handler(event, context):
    """
    Lambda 핸들러 (Function URL 호환)

    Actions:
        - start: EC2 시작 후 running 상태까지 대기
        - stop: EC2 종료
        - status: 현재 상태 조회
    """
    import json

    # Function URL인 경우 body에서 파싱
    if 'body' in event:
        try:
            body = json.loads(event['body']) if isinstance(event['body'], str) else event['body']
            action = body.get('action', 'status')
        except:
            action = 'status'
    else:
        # 직접 호출인 경우
        action = event.get('action', 'status')

    try:
        if action == 'start':
            return start_instance()
        elif action == 'stop':
            return stop_instance()
        elif action == 'status':
            return get_status()
        else:
            return {'error': f'Unknown action: {action}'}
    except Exception as e:
        return {'error': str(e)}


def get_instance_info():
    """인스턴스 정보 조회 (내부용)"""
    response = ec2.describe_instances(InstanceIds=[PIPELINE_INSTANCE_ID])
    instance = response['Reservations'][0]['Instances'][0]
    return {
        'instance_id': PIPELINE_INSTANCE_ID,
        'state': instance['State']['Name'],
        'public_ip': instance.get('PublicIpAddress')
    }


def get_status():
    """인스턴스 상태 조회"""
    return get_instance_info()


def start_instance():
    """인스턴스 시작"""
    info = get_instance_info()

    if info['state'] == 'running':
        return {
            'message': 'Instance already running',
            **info
        }

    if info['state'] not in ['stopped', 'stopping']:
        return {
            'error': f"Cannot start instance in state: {info['state']}",
            **info
        }

    # 인스턴스 시작
    ec2.start_instances(InstanceIds=[PIPELINE_INSTANCE_ID])

    # running 상태까지 대기 (최대 60초)
    for _ in range(30):
        time.sleep(2)
        info = get_instance_info()
        if info['state'] == 'running':
            time.sleep(5)  # IP 할당 대기
            info = get_instance_info()
            return {
                'message': 'Instance started successfully',
                **info
            }

    return {
        'message': 'Instance starting (may take a moment)',
        'instance_id': PIPELINE_INSTANCE_ID,
        'state': 'pending'
    }


def stop_instance():
    """인스턴스 종료"""
    info = get_instance_info()

    if info['state'] == 'stopped':
        return {
            'message': 'Instance already stopped',
            **info
        }

    if info['state'] != 'running':
        return {
            'error': f"Cannot stop instance in state: {info['state']}",
            **info
        }

    # 인스턴스 종료
    ec2.stop_instances(InstanceIds=[PIPELINE_INSTANCE_ID])

    return {
        'message': 'Instance stopping',
        'instance_id': PIPELINE_INSTANCE_ID,
        'state': 'stopping'
    }
