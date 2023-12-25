
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001700 <r_2_2_3n1>:
	s_clause 0x1                                               // 000000001700: BF850001
	s_load_b128 s[4:7], s[0:1], null                           // 000000001704: F4080100 F8000000
	s_load_b64 s[0:1], s[0:1], 0x10                            // 00000000170C: F4040000 F8000010
	s_mul_i32 s2, s15, 5                                       // 000000001714: 9602850F
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001718: BF870499
	s_ashr_i32 s3, s2, 31                                      // 00000000171C: 86039F02
	s_lshl_b64 s[2:3], s[2:3], 2                               // 000000001720: 84828202
	s_waitcnt lgkmcnt(0)                                       // 000000001724: BF89FC07
	s_add_u32 s6, s6, s2                                       // 000000001728: 80060206
	s_addc_u32 s7, s7, s3                                      // 00000000172C: 82070307
	s_lshl_b32 s2, s14, 1                                      // 000000001730: 8402810E
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001734: BF870499
	s_ashr_i32 s3, s2, 31                                      // 000000001738: 86039F02
	s_lshl_b64 s[2:3], s[2:3], 2                               // 00000000173C: 84828202
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001740: BF870009
	s_add_u32 s2, s6, s2                                       // 000000001744: 80020206
	s_addc_u32 s3, s7, s3                                      // 000000001748: 82030307
	s_load_b64 s[6:7], s[2:3], null                            // 00000000174C: F4040181 F8000000
	s_clause 0x1                                               // 000000001754: BF850001
	s_load_b64 s[8:9], s[0:1], null                            // 000000001758: F4040200 F8000000
	s_load_b32 s10, s[0:1], 0x8                                // 000000001760: F4000280 F8000008
	s_load_b32 s2, s[2:3], 0x8                                 // 000000001768: F4000081 F8000008
	s_lshl_b32 s0, s15, 1                                      // 000000001770: 8400810F
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001774: BF870499
	s_ashr_i32 s1, s0, 31                                      // 000000001778: 86019F00
	s_lshl_b64 s[0:1], s[0:1], 2                               // 00000000177C: 84808200
	s_waitcnt lgkmcnt(0)                                       // 000000001780: BF89FC07
	v_fma_f32 v0, s6, s8, 0                                    // 000000001784: D6130000 02001006
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000178C: BF870091
	v_fmac_f32_e64 v0, s7, s9                                  // 000000001790: D52B0000 00001207
	v_fmac_f32_e64 v0, s2, s10                                 // 000000001798: D52B0000 00001402
	s_add_u32 s2, s4, s0                                       // 0000000017A0: 80020004
	s_addc_u32 s3, s5, s1                                      // 0000000017A4: 82030105
	s_ashr_i32 s15, s14, 31                                    // 0000000017A8: 860F9F0E
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)// 0000000017AC: BF8704A1
	v_dual_mov_b32 v1, 0 :: v_dual_max_f32 v0, 0, v0           // 0000000017B0: CA140080 01000080
	s_lshl_b64 s[0:1], s[14:15], 2                             // 0000000017B8: 8480820E
	s_add_u32 s0, s2, s0                                       // 0000000017BC: 80000002
	s_addc_u32 s1, s3, s1                                      // 0000000017C0: 82010103
	global_store_b32 v1, v0, s[0:1]                            // 0000000017C4: DC6A0000 00000001
	s_nop 0                                                    // 0000000017CC: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 0000000017D0: BFB60003
	s_endpgm                                                   // 0000000017D4: BFB00000
