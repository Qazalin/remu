
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001700 <r_4_5_7_3_3_3_3_3>:
	s_mul_hi_i32 s2, s13, 0x55555556                           // 000000001700: 9702FF0D 55555556
	s_load_b128 s[4:7], s[0:1], null                           // 000000001708: F4080100 F8000000
	s_lshr_b32 s3, s2, 31                                      // 000000001710: 85039F02
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_4) | instid1(SALU_CYCLE_1)// 000000001714: BF8704D9
	s_add_i32 s8, s2, s3                                       // 000000001718: 81080302
	s_load_b64 s[2:3], s[0:1], 0x10                            // 00000000171C: F4040080 F8000010
	s_mul_hi_i32 s9, s8, 0x55555556                            // 000000001724: 9709FF08 55555556
	s_mul_hi_i32 s1, s13, 0x38e38e39                           // 00000000172C: 9701FF0D 38E38E39
	s_lshr_b32 s0, s9, 31                                      // 000000001734: 85009F09
	s_add_i32 s0, s9, s0                                       // 000000001738: 81000009
	s_mul_i32 s9, s8, 3                                        // 00000000173C: 96098308
	s_mul_i32 s10, s0, 3                                       // 000000001740: 960A8300
	s_sub_i32 s0, s13, s9                                      // 000000001744: 8180090D
	s_sub_i32 s33, s8, s10                                     // 000000001748: 81A10A08
	s_mul_i32 s8, s15, 0x177                                   // 00000000174C: 9608FF0F 00000177
	s_lshr_b32 s10, s1, 31                                     // 000000001754: 850A9F01
	s_ashr_i32 s9, s8, 31                                      // 000000001758: 86099F08
	s_ashr_i32 s1, s1, 1                                       // 00000000175C: 86018101
	s_lshl_b64 s[8:9], s[8:9], 2                               // 000000001760: 84888208
	s_add_i32 s34, s1, s10                                     // 000000001764: 81220A01
	s_waitcnt lgkmcnt(0)                                       // 000000001768: BF89FC07
	s_add_u32 s1, s6, s8                                       // 00000000176C: 80010806
	s_mul_i32 s6, s14, 0x4b                                    // 000000001770: 9606FF0E 0000004B
	s_addc_u32 s8, s7, s9                                      // 000000001778: 82080907
	s_ashr_i32 s7, s6, 31                                      // 00000000177C: 86079F06
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001780: BF870499
	s_lshl_b64 s[6:7], s[6:7], 2                               // 000000001784: 84868206
	s_add_u32 s1, s1, s6                                       // 000000001788: 80010601
	s_mul_i32 s6, s33, 5                                       // 00000000178C: 96068521
	s_addc_u32 s8, s8, s7                                      // 000000001790: 82080708
	s_ashr_i32 s7, s6, 31                                      // 000000001794: 86079F06
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001798: BF870499
	s_lshl_b64 s[6:7], s[6:7], 2                               // 00000000179C: 84868206
	s_add_u32 s6, s1, s6                                       // 0000000017A0: 80060601
	s_addc_u32 s7, s8, s7                                      // 0000000017A4: 82070708
	s_ashr_i32 s1, s0, 31                                      // 0000000017A8: 86019F00
	s_mul_i32 s8, s14, 0xbd                                    // 0000000017AC: 9608FF0E 000000BD
	s_lshl_b64 s[0:1], s[0:1], 2                               // 0000000017B4: 84808200
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)// 0000000017B8: BF8704B9
	s_add_u32 s6, s6, s0                                       // 0000000017BC: 80060006
	s_addc_u32 s7, s7, s1                                      // 0000000017C0: 82070107
	s_ashr_i32 s9, s8, 31                                      // 0000000017C4: 86099F08
	s_lshl_b64 s[8:9], s[8:9], 2                               // 0000000017C8: 84888208
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_3) | instid1(SALU_CYCLE_1)// 0000000017CC: BF8704C9
	s_add_u32 s8, s2, s8                                       // 0000000017D0: 80080802
	s_mul_i32 s2, s34, 27                                      // 0000000017D4: 96029B22
	s_addc_u32 s9, s3, s9                                      // 0000000017D8: 82090903
	s_ashr_i32 s3, s2, 31                                      // 0000000017DC: 86039F02
	s_lshl_b64 s[2:3], s[2:3], 2                               // 0000000017E0: 84828202
	s_delay_alu instid0(SALU_CYCLE_1)                          // 0000000017E4: BF870009
	s_add_u32 s2, s8, s2                                       // 0000000017E8: 80020208
	s_addc_u32 s3, s9, s3                                      // 0000000017EC: 82030309
	s_load_b512 s[16:31], s[2:3], null                         // 0000000017F0: F4100401 F8000000
	s_clause 0x3                                               // 0000000017F8: BF850003
	s_load_b64 s[8:9], s[6:7], null                            // 0000000017FC: F4040203 F8000000
	s_load_b32 s12, s[6:7], 0x8                                // 000000001804: F4000303 F8000008
	s_load_b64 s[10:11], s[6:7], 0x14                          // 00000000180C: F4040283 F8000014
	s_load_b32 s13, s[6:7], 0x1c                               // 000000001814: F4000343 F800001C
	s_waitcnt lgkmcnt(0)                                       // 00000000181C: BF89FC07
	v_fma_f32 v0, s8, s16, 0                                   // 000000001820: D6130000 02002008
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001828: BF8700A1
	v_fmac_f32_e64 v0, s9, s17                                 // 00000000182C: D52B0000 00002209
	s_load_b64 s[8:9], s[6:7], 0x28                            // 000000001834: F4040203 F8000028
	v_fmac_f32_e64 v0, s12, s18                                // 00000000183C: D52B0000 0000240C
	s_load_b32 s12, s[6:7], 0x30                               // 000000001844: F4000303 F8000030
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000184C: BF870091
	v_fmac_f32_e64 v0, s10, s19                                // 000000001850: D52B0000 0000260A
	v_fmac_f32_e64 v0, s11, s20                                // 000000001858: D52B0000 0000280B
	s_load_b64 s[10:11], s[6:7], 0x64                          // 000000001860: F4040283 F8000064
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)// 000000001868: BF8700B1
	v_fmac_f32_e64 v0, s13, s21                                // 00000000186C: D52B0000 00002A0D
	s_load_b32 s13, s[6:7], 0x6c                               // 000000001874: F4000343 F800006C
	s_waitcnt lgkmcnt(0)                                       // 00000000187C: BF89FC07
	v_fmac_f32_e64 v0, s8, s22                                 // 000000001880: D52B0000 00002C08
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001888: BF8700A1
	v_fmac_f32_e64 v0, s9, s23                                 // 00000000188C: D52B0000 00002E09
	s_load_b64 s[8:9], s[6:7], 0x7c                            // 000000001894: F4040203 F800007C
	v_fmac_f32_e64 v0, s12, s24                                // 00000000189C: D52B0000 0000300C
	s_load_b32 s12, s[6:7], 0x78                               // 0000000018A4: F4000303 F8000078
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000018AC: BF870091
	v_fmac_f32_e64 v0, s10, s25                                // 0000000018B0: D52B0000 0000320A
	v_fmac_f32_e64 v0, s11, s26                                // 0000000018B8: D52B0000 0000340B
	s_clause 0x1                                               // 0000000018C0: BF850001
	s_load_b32 s26, s[6:7], 0x8c                               // 0000000018C4: F4000683 F800008C
	s_load_b64 s[10:11], s[6:7], 0x90                          // 0000000018CC: F4040283 F8000090
	s_load_b256 s[16:23], s[2:3], 0x40                         // 0000000018D4: F40C0401 F8000040
	v_fmac_f32_e64 v0, s13, s27                                // 0000000018DC: D52B0000 0000360D
	s_waitcnt lgkmcnt(0)                                       // 0000000018E4: BF89FC07
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000018E8: BF870001
	v_fmac_f32_e64 v0, s12, s28                                // 0000000018EC: D52B0000 0000380C
	s_clause 0x2                                               // 0000000018F4: BF850002
	s_load_b32 s27, s[6:7], 0xdc                               // 0000000018F8: F40006C3 F80000DC
	s_load_b64 s[12:13], s[6:7], 0xe0                          // 000000001900: F4040303 F80000E0
	s_load_b64 s[24:25], s[6:7], 0xc8                          // 000000001908: F4040603 F80000C8
	v_fmac_f32_e64 v0, s8, s29                                 // 000000001910: D52B0000 00003A08
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001918: BF8700A1
	v_fmac_f32_e64 v0, s9, s30                                 // 00000000191C: D52B0000 00003C09
	s_load_b64 s[8:9], s[2:3], 0x60                            // 000000001924: F4040201 F8000060
	v_fmac_f32_e64 v0, s26, s31                                // 00000000192C: D52B0000 00003E1A
	s_load_b32 s26, s[6:7], 0xd0                               // 000000001934: F4000683 F80000D0
	s_delay_alu instid0(VALU_DEP_1)                            // 00000000193C: BF870001
	v_fmac_f32_e64 v0, s10, s16                                // 000000001940: D52B0000 0000200A
	s_clause 0x1                                               // 000000001948: BF850001
	s_load_b32 s16, s[6:7], 0xf0                               // 00000000194C: F4000403 F80000F0
	s_load_b64 s[6:7], s[6:7], 0xf4                            // 000000001954: F4040183 F80000F4
	s_mul_i32 s10, s15, 0x13b                                  // 00000000195C: 960AFF0F 0000013B
	s_load_b32 s15, s[2:3], 0x68                               // 000000001964: F40003C1 F8000068
	v_fmac_f32_e64 v0, s11, s17                                // 00000000196C: D52B0000 0000220B
	s_ashr_i32 s11, s10, 31                                    // 000000001974: 860B9F0A
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)// 000000001978: BF8700C9
	s_lshl_b64 s[2:3], s[10:11], 2                             // 00000000197C: 8482820A
	s_mul_i32 s10, s14, 63                                     // 000000001980: 960ABF0E
	s_waitcnt lgkmcnt(0)                                       // 000000001984: BF89FC07
	v_fmac_f32_e64 v0, s24, s18                                // 000000001988: D52B0000 00002418
	v_fmac_f32_e64 v0, s25, s19                                // 000000001990: D52B0000 00002619
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001998: BF870091
	v_fmac_f32_e64 v0, s26, s20                                // 00000000199C: D52B0000 0000281A
	v_fmac_f32_e64 v0, s27, s21                                // 0000000019A4: D52B0000 00002A1B
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000019AC: BF870001
	v_fmac_f32_e64 v0, s12, s22                                // 0000000019B0: D52B0000 00002C0C
	s_add_u32 s12, s4, s2                                      // 0000000019B8: 800C0204
	s_addc_u32 s5, s5, s3                                      // 0000000019BC: 82050305
	s_ashr_i32 s11, s10, 31                                    // 0000000019C0: 860B9F0A
	s_mul_i32 s4, s34, 9                                       // 0000000019C4: 96048922
	v_fmac_f32_e64 v0, s13, s23                                // 0000000019C8: D52B0000 00002E0D
	s_lshl_b64 s[2:3], s[10:11], 2                             // 0000000019D0: 8482820A
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)// 0000000019D4: BF8700C1
	v_fmac_f32_e64 v0, s16, s8                                 // 0000000019D8: D52B0000 00001010
	s_add_u32 s8, s12, s2                                      // 0000000019E0: 8008020C
	s_addc_u32 s10, s5, s3                                     // 0000000019E4: 820A0305
	s_ashr_i32 s5, s4, 31                                      // 0000000019E8: 86059F04
	v_fmac_f32_e64 v0, s6, s9                                  // 0000000019EC: D52B0000 00001206
	s_lshl_b64 s[2:3], s[4:5], 2                               // 0000000019F4: 84828204
	s_mul_i32 s4, s33, 3                                       // 0000000019F8: 96048321
	s_add_u32 s6, s8, s2                                       // 0000000019FC: 80060208
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001A00: BF870001
	v_fmac_f32_e64 v0, s7, s15                                 // 000000001A04: D52B0000 00001E07
	s_addc_u32 s7, s10, s3                                     // 000000001A0C: 8207030A
	s_ashr_i32 s5, s4, 31                                      // 000000001A10: 86059F04
	v_mov_b32_e32 v1, 0                                        // 000000001A14: 7E020280
	s_lshl_b64 s[2:3], s[4:5], 2                               // 000000001A18: 84828204
	v_max_f32_e32 v0, 0, v0                                    // 000000001A1C: 20000080
	s_add_u32 s2, s6, s2                                       // 000000001A20: 80020206
	s_addc_u32 s3, s7, s3                                      // 000000001A24: 82030307
	s_add_u32 s0, s2, s0                                       // 000000001A28: 80000002
	s_addc_u32 s1, s3, s1                                      // 000000001A2C: 82010103
	global_store_b32 v1, v0, s[0:1]                            // 000000001A30: DC6A0000 00000001
	s_nop 0                                                    // 000000001A38: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 000000001A3C: BFB60003
	s_endpgm                                                   // 000000001A40: BFB00000
