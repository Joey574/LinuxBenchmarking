start_time=$(date +%s.%N)

# build directory
mkdir -p build
cd build

export OMP_NUM_THREADS=$(nproc)
export OMP_PLACES=cores
export OMP_PROC_BIND=close

# configure
cmake .. -DCMAKE_BUILD_TYPE="$build_type" -G Ninja

# build program
echo "-- Building program ($build_type)"
cmake --build . -j

# output
cd ../
end_time=$(date +%s.%N)
elapsed=$(echo "$end_time - $start_time" | bc)

printf "Build completed in %.2f seconds\n" "$elapsed"
