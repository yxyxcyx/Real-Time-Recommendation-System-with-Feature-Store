#!/usr/bin/env python3
"""Performance testing script for RecSys API."""

import asyncio
import json
import time
import statistics
import concurrent.futures
import argparse
from typing import List, Dict, Any
import requests
import numpy as np


class PerformanceTester:
    """Performance testing framework for RecSys."""
    
    def __init__(self, base_url: str = "http://localhost:8000", endpoint: str = "/recommend_mock"):
        self.base_url = base_url
        self.endpoint = endpoint
        self.session = requests.Session()
        
    def health_check(self) -> bool:
        """Check if API is healthy."""
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def test_recommendation_endpoint(self, user_id: str, num_recommendations: int = 5) -> Dict[str, Any]:
        """Test single recommendation request."""
        start_time = time.time()
        
        try:
            payload = {
                "user_id": user_id,
                "num_recommendations": num_recommendations
            }
            
            response = self.session.post(
                f"{self.base_url}{self.endpoint}",
                json=payload,
                timeout=30
            )
            
            end_time = time.time()
            latency_ms = (end_time - start_time) * 1000
            
            return {
                "success": response.status_code == 200,
                "status_code": response.status_code,
                "latency_ms": latency_ms,
                "response_size": len(response.content),
                "user_id": user_id
            }
            
        except Exception as e:
            end_time = time.time()
            latency_ms = (end_time - start_time) * 1000
            
            return {
                "success": False,
                "error": str(e),
                "latency_ms": latency_ms,
                "user_id": user_id
            }
    
    def load_test(self, concurrent_users: int = 10, requests_per_user: int = 100) -> Dict[str, Any]:
        """Run load test with concurrent users."""
        print(f"Starting load test: {concurrent_users} users, {requests_per_user} requests each")
        
        def user_session(user_id: int) -> List[Dict[str, Any]]:
            """Simulate a user session."""
            results = []
            for req_id in range(requests_per_user):
                result = self.test_recommendation_endpoint(f"user_{user_id}_{req_id}")
                results.append(result)
                
                # Small delay between requests
                time.sleep(0.01)
            
            return results
        
        # Run concurrent user sessions
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_users) as executor:
            futures = [executor.submit(user_session, user_id) for user_id in range(concurrent_users)]
            all_results = []
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    user_results = future.result()
                    all_results.extend(user_results)
                except Exception as e:
                    print(f"User session failed: {e}")
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Calculate statistics
        successful_requests = [r for r in all_results if r["success"]]
        failed_requests = [r for r in all_results if not r["success"]]
        
        if successful_requests:
            latencies = [r["latency_ms"] for r in successful_requests]
            
            stats = {
                "total_requests": len(all_results),
                "successful_requests": len(successful_requests),
                "failed_requests": len(failed_requests),
                "success_rate": len(successful_requests) / len(all_results) * 100,
                "total_time_seconds": total_time,
                "requests_per_second": len(all_results) / total_time,
                "latency_stats": {
                    "avg_ms": statistics.mean(latencies),
                    "median_ms": statistics.median(latencies),
                    "min_ms": min(latencies),
                    "max_ms": max(latencies),
                    "p95_ms": np.percentile(latencies, 95),
                    "p99_ms": np.percentile(latencies, 99)
                }
            }
        else:
            stats = {
                "total_requests": len(all_results),
                "successful_requests": 0,
                "failed_requests": len(failed_requests),
                "success_rate": 0,
                "error": "All requests failed"
            }
        
        return stats
    
    def stress_test(self, duration_seconds: int = 60, max_concurrent: int = 50) -> Dict[str, Any]:
        """Run stress test for specified duration."""
        print(f"Starting stress test for {duration_seconds} seconds with up to {max_concurrent} concurrent users")
        
        results = []
        start_time = time.time()
        request_id = 0
        
        def make_request():
            nonlocal request_id
            result = self.test_recommendation_endpoint(f"stress_user_{request_id}")
            results.append(result)
            request_id += 1
        
        # Gradually increase load
        futures = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            while time.time() - start_time < duration_seconds:
                # Submit requests at increasing rate
                for _ in range(min(10, max_concurrent - len(futures))):
                    future = executor.submit(make_request)
                    futures.append(future)
                
                # Clean up completed futures
                futures = [f for f in futures if not f.done()]
                time.sleep(0.1)
            
            # Wait for remaining requests
            concurrent.futures.wait(futures)
        
        # Calculate statistics
        successful_requests = [r for r in results if r["success"]]
        failed_requests = [r for r in results if not r["success"]]
        
        if successful_requests:
            latencies = [r["latency_ms"] for r in successful_requests]
            
            stats = {
                "duration_seconds": duration_seconds,
                "total_requests": len(results),
                "successful_requests": len(successful_requests),
                "failed_requests": len(failed_requests),
                "success_rate": len(successful_requests) / len(results) * 100,
                "requests_per_second": len(results) / duration_seconds,
                "latency_stats": {
                    "avg_ms": statistics.mean(latencies),
                    "median_ms": statistics.median(latencies),
                    "min_ms": min(latencies),
                    "max_ms": max(latencies),
                    "p95_ms": np.percentile(latencies, 95),
                    "p99_ms": np.percentile(latencies, 99)
                }
            }
        else:
            stats = {
                "duration_seconds": duration_seconds,
                "total_requests": len(results),
                "successful_requests": 0,
                "failed_requests": len(failed_requests),
                "success_rate": 0,
                "error": "All requests failed"
            }
        
        return stats
    
    def print_results(self, results: Dict[str, Any], test_name: str):
        """Print test results in a readable format."""
        print(f"\n{'='*60}")
        print(f"PERFORMANCE TEST RESULTS: {test_name}")
        print(f"{'='*60}")
        
        if "error" in results:
            print(f"TEST FAILED: {results['error']}")
            return
        
        print(f"Total Requests: {results['total_requests']}")
        print(f"Successful: {results['successful_requests']}")
        print(f"Failed: {results['failed_requests']}")
        print(f"Success Rate: {results['success_rate']:.2f}%")
        
        if "requests_per_second" in results:
            print(f"Requests/Second: {results['requests_per_second']:.2f}")
        
        if "total_time_seconds" in results:
            print(f"Total Time: {results['total_time_seconds']:.2f}s")
        
        if "latency_stats" in results:
            latency = results["latency_stats"]
            print(f"\nLatency Statistics:")
            print(f"   Average: {latency['avg_ms']:.2f}ms")
            print(f"   Median:  {latency['median_ms']:.2f}ms")
            print(f"   Min:     {latency['min_ms']:.2f}ms")
            print(f"   Max:     {latency['max_ms']:.2f}ms")
            print(f"   95th:    {latency['p95_ms']:.2f}ms")
            print(f"   99th:    {latency['p99_ms']:.2f}ms")
        
        print(f"{'='*60}\n")


def main():
    """Main performance testing function."""
    parser = argparse.ArgumentParser(description="RecSys Performance Testing")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--endpoint", default="/recommend_mock", help="Recommendation endpoint to test")
    parser.add_argument("--test", choices=["load", "stress", "all"], default="all", help="Test type to run")
    parser.add_argument("--users", type=int, default=10, help="Concurrent users for load test")
    parser.add_argument("--requests", type=int, default=100, help="Requests per user for load test")
    parser.add_argument("--duration", type=int, default=60, help="Duration for stress test (seconds)")
    
    args = parser.parse_args()
    
    tester = PerformanceTester(args.url, args.endpoint)
    
    # Health check
    print("Performing health check...")
    if not tester.health_check():
        print("API health check failed. Please ensure the API is running.")
        return
    
    print("API is healthy")
    
    # Run tests
    if args.test in ["load", "all"]:
        print(f"\nRunning load test...")
        load_results = tester.load_test(args.users, args.requests)
        tester.print_results(load_results, "Load Test")
        
        # Performance assessment
        if load_results.get("success_rate", 0) > 95:
            if load_results["latency_stats"]["avg_ms"] < 100:
                print("EXCELLENT: High success rate and low latency!")
            elif load_results["latency_stats"]["avg_ms"] < 500:
                print("GOOD: High success rate with acceptable latency")
            else:
                print("NEEDS IMPROVEMENT: High success rate but high latency")
        else:
            print("POOR: Low success rate needs immediate attention")
    
    if args.test in ["stress", "all"]:
        print(f"\nRunning stress test...")
        stress_results = tester.stress_test(args.duration)
        tester.print_results(stress_results, "Stress Test")
        
        # Performance assessment
        if stress_results.get("success_rate", 0) > 95:
            print("EXCELLENT: System handles stress well!")
        elif stress_results.get("success_rate", 0) > 80:
            print("ACCEPTABLE: System degrades under stress but recovers")
        else:
            print("POOR: System cannot handle stress load")


if __name__ == "__main__":
    main()
