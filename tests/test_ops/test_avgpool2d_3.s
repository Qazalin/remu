
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001700 <r_64_22_5_5_5>:
	s_load_b128 s[0:3], s[0:1], null                           // 000000001700: F4080000 F8000000
	s_mul_i32 s4, s15, 0xc24                                   // 000000001708: 9604FF0F 00000C24
	s_mul_i32 s6, s14, 0x8c                                    // 000000001710: 9606FF0E 0000008C
	s_ashr_i32 s5, s4, 31                                      // 000000001718: 86059F04
	s_mov_b32 s12, s13                                         // 00000000171C: BE8C000D
	s_lshl_b64 s[4:5], s[4:5], 2                               // 000000001720: 84848204
	v_mov_b32_e32 v1, 0                                        // 000000001724: 7E020280
	s_waitcnt lgkmcnt(0)                                       // 000000001728: BF89FC07
	s_add_u32 s8, s2, s4                                       // 00000000172C: 80080402
	s_addc_u32 s5, s3, s5                                      // 000000001730: 82050503
	s_ashr_i32 s7, s6, 31                                      // 000000001734: 86079F06
	s_mul_i32 s4, s13, 5                                       // 000000001738: 9604850D
	s_lshl_b64 s[2:3], s[6:7], 2                               // 00000000173C: 84828206
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)// 000000001740: BF8704B9
	s_add_u32 s6, s8, s2                                       // 000000001744: 80060208
	s_addc_u32 s7, s5, s3                                      // 000000001748: 82070305
	s_ashr_i32 s5, s4, 31                                      // 00000000174C: 86059F04
	s_lshl_b64 s[2:3], s[4:5], 2                               // 000000001750: 84828204
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001754: BF870009
	s_add_u32 s2, s6, s2                                       // 000000001758: 80020206
	s_addc_u32 s3, s7, s3                                      // 00000000175C: 82030307
	s_clause 0x2                                               // 000000001760: BF850002
	s_load_b128 s[4:7], s[2:3], null                           // 000000001764: F4080101 F8000000
	s_load_b32 s13, s[2:3], 0x10                               // 00000000176C: F4000341 F8000010
	s_load_b128 s[8:11], s[2:3], 0x70                          // 000000001774: F4080201 F8000070
	s_waitcnt lgkmcnt(0)                                       // 00000000177C: BF89FC07
	v_add_f32_e64 v0, s4, 0                                    // 000000001780: D5030000 00010004
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001788: BF870091
	v_add_f32_e32 v0, s5, v0                                   // 00000000178C: 06000005
	v_add_f32_e32 v0, s6, v0                                   // 000000001790: 06000006
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001794: BF8700A1
	v_add_f32_e32 v0, s7, v0                                   // 000000001798: 06000007
	s_load_b128 s[4:7], s[2:3], 0xe0                           // 00000000179C: F4080101 F80000E0
	v_add_f32_e32 v0, s13, v0                                  // 0000000017A4: 0600000D
	s_load_b32 s13, s[2:3], 0x80                               // 0000000017A8: F4000341 F8000080
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000017B0: BF870091
	v_add_f32_e32 v0, s8, v0                                   // 0000000017B4: 06000008
	v_add_f32_e32 v0, s9, v0                                   // 0000000017B8: 06000009
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000017BC: BF870091
	v_add_f32_e32 v0, s10, v0                                  // 0000000017C0: 0600000A
	v_add_f32_e32 v0, s11, v0                                  // 0000000017C4: 0600000B
	s_load_b128 s[8:11], s[2:3], 0x150                         // 0000000017C8: F4080201 F8000150
	s_waitcnt lgkmcnt(0)                                       // 0000000017D0: BF89FC07
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 0000000017D4: BF8700A1
	v_add_f32_e32 v0, s13, v0                                  // 0000000017D8: 0600000D
	s_load_b32 s13, s[2:3], 0xf0                               // 0000000017DC: F4000341 F80000F0
	v_add_f32_e32 v0, s4, v0                                   // 0000000017E4: 06000004
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000017E8: BF870091
	v_add_f32_e32 v0, s5, v0                                   // 0000000017EC: 06000005
	v_add_f32_e32 v0, s6, v0                                   // 0000000017F0: 06000006
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)// 0000000017F4: BF8700B1
	v_add_f32_e32 v0, s7, v0                                   // 0000000017F8: 06000007
	s_load_b128 s[4:7], s[2:3], 0x1c0                          // 0000000017FC: F4080101 F80001C0
	s_waitcnt lgkmcnt(0)                                       // 000000001804: BF89FC07
	v_add_f32_e32 v0, s13, v0                                  // 000000001808: 0600000D
	s_load_b32 s13, s[2:3], 0x160                              // 00000000180C: F4000341 F8000160
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)// 000000001814: BF8704B1
	v_add_f32_e32 v0, s8, v0                                   // 000000001818: 06000008
	s_load_b32 s8, s[2:3], 0x1d0                               // 00000000181C: F4000201 F80001D0
	s_mul_i32 s2, s15, 0x6e                                    // 000000001824: 9602FF0F 0000006E
	s_ashr_i32 s3, s2, 31                                      // 00000000182C: 86039F02
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)// 000000001830: BF8704A1
	v_add_f32_e32 v0, s9, v0                                   // 000000001834: 06000009
	s_lshl_b64 s[2:3], s[2:3], 2                               // 000000001838: 84828202
	s_add_u32 s2, s0, s2                                       // 00000000183C: 80020200
	s_addc_u32 s3, s1, s3                                      // 000000001840: 82030301
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001844: BF870091
	v_add_f32_e32 v0, s10, v0                                  // 000000001848: 0600000A
	v_add_f32_e32 v0, s11, v0                                  // 00000000184C: 0600000B
	s_waitcnt lgkmcnt(0)                                       // 000000001850: BF89FC07
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001854: BF870091
	v_add_f32_e32 v0, s13, v0                                  // 000000001858: 0600000D
	v_add_f32_e32 v0, s4, v0                                   // 00000000185C: 06000004
	s_mul_i32 s4, s14, 5                                       // 000000001860: 9604850E
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)// 000000001864: BF8704A1
	v_add_f32_e32 v0, s5, v0                                   // 000000001868: 06000005
	s_ashr_i32 s5, s4, 31                                      // 00000000186C: 86059F04
	s_lshl_b64 s[0:1], s[4:5], 2                               // 000000001870: 84808204
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)// 000000001874: BF8700C1
	v_add_f32_e32 v0, s6, v0                                   // 000000001878: 06000006
	s_add_u32 s2, s2, s0                                       // 00000000187C: 80020002
	s_addc_u32 s3, s3, s1                                      // 000000001880: 82030103
	s_ashr_i32 s13, s12, 31                                    // 000000001884: 860D9F0C
	v_add_f32_e32 v0, s7, v0                                   // 000000001888: 06000007
	s_lshl_b64 s[0:1], s[12:13], 2                             // 00000000188C: 8480820C
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001890: BF8700A9
	s_add_u32 s0, s2, s0                                       // 000000001894: 80000002
	s_addc_u32 s1, s3, s1                                      // 000000001898: 82010103
	v_add_f32_e32 v0, s8, v0                                   // 00000000189C: 06000008
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000018A0: BF870001
	v_mul_f32_e32 v0, 0x3d23d70a, v0                           // 0000000018A4: 100000FF 3D23D70A
	global_store_b32 v1, v0, s[0:1]                            // 0000000018AC: DC6A0000 00000001
	s_nop 0                                                    // 0000000018B4: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 0000000018B8: BFB60003
	s_endpgm                                                   // 0000000018BC: BFB00000
