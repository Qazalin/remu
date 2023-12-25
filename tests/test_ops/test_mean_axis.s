
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001700 <r_3_6_20>:
	s_load_b128 s[4:7], s[0:1], null                           // 000000001700: F4080100 F8000000
	s_mul_i32 s0, s15, 0x78                                    // 000000001708: 9600FF0F 00000078
	s_mov_b32 s2, s15                                          // 000000001710: BE82000F
	s_ashr_i32 s1, s0, 31                                      // 000000001714: 86019F00
	s_mul_i32 s2, s2, 6                                        // 000000001718: 96028602
	s_lshl_b64 s[0:1], s[0:1], 2                               // 00000000171C: 84808200
	v_mov_b32_e32 v1, 0                                        // 000000001720: 7E020280
	s_waitcnt lgkmcnt(0)                                       // 000000001724: BF89FC07
	s_add_u32 s3, s6, s0                                       // 000000001728: 80030006
	s_addc_u32 s7, s7, s1                                      // 00000000172C: 82070107
	s_ashr_i32 s15, s14, 31                                    // 000000001730: 860F9F0E
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001734: BF870499
	s_lshl_b64 s[0:1], s[14:15], 2                             // 000000001738: 8480820E
	s_add_u32 s6, s3, s0                                       // 00000000173C: 80060003
	s_addc_u32 s7, s7, s1                                      // 000000001740: 82070107
	s_clause 0x7                                               // 000000001744: BF850007
	s_load_b32 s3, s[6:7], null                                // 000000001748: F40000C3 F8000000
	s_load_b32 s8, s[6:7], 0x18                                // 000000001750: F4000203 F8000018
	s_load_b32 s9, s[6:7], 0x30                                // 000000001758: F4000243 F8000030
	s_load_b32 s10, s[6:7], 0x48                               // 000000001760: F4000283 F8000048
	s_load_b32 s11, s[6:7], 0x60                               // 000000001768: F40002C3 F8000060
	s_load_b32 s12, s[6:7], 0x78                               // 000000001770: F4000303 F8000078
	s_load_b32 s13, s[6:7], 0x90                               // 000000001778: F4000343 F8000090
	s_load_b32 s14, s[6:7], 0xa8                               // 000000001780: F4000383 F80000A8
	s_waitcnt lgkmcnt(0)                                       // 000000001788: BF89FC07
	v_add_f32_e64 v0, s3, 0                                    // 00000000178C: D5030000 00010003
	s_load_b32 s3, s[6:7], 0xc0                                // 000000001794: F40000C3 F80000C0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 00000000179C: BF8700A1
	v_add_f32_e32 v0, s8, v0                                   // 0000000017A0: 06000008
	s_load_b32 s8, s[6:7], 0xd8                                // 0000000017A4: F4000203 F80000D8
	v_add_f32_e32 v0, s9, v0                                   // 0000000017AC: 06000009
	s_load_b32 s9, s[6:7], 0xf0                                // 0000000017B0: F4000243 F80000F0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000017B8: BF870091
	v_add_f32_e32 v0, s10, v0                                  // 0000000017BC: 0600000A
	v_add_f32_e32 v0, s11, v0                                  // 0000000017C0: 0600000B
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000017C4: BF870091
	v_add_f32_e32 v0, s12, v0                                  // 0000000017C8: 0600000C
	v_add_f32_e32 v0, s13, v0                                  // 0000000017CC: 0600000D
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000017D0: BF870001
	v_add_f32_e32 v0, s14, v0                                  // 0000000017D4: 0600000E
	s_clause 0x4                                               // 0000000017D8: BF850004
	s_load_b32 s10, s[6:7], 0x108                              // 0000000017DC: F4000283 F8000108
	s_load_b32 s11, s[6:7], 0x120                              // 0000000017E4: F40002C3 F8000120
	s_load_b32 s12, s[6:7], 0x138                              // 0000000017EC: F4000303 F8000138
	s_load_b32 s13, s[6:7], 0x150                              // 0000000017F4: F4000343 F8000150
	s_load_b32 s14, s[6:7], 0x168                              // 0000000017FC: F4000383 F8000168
	s_waitcnt lgkmcnt(0)                                       // 000000001804: BF89FC07
	v_add_f32_e32 v0, s3, v0                                   // 000000001808: 06000003
	s_load_b32 s3, s[6:7], 0x180                               // 00000000180C: F40000C3 F8000180
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001814: BF8700A1
	v_add_f32_e32 v0, s8, v0                                   // 000000001818: 06000008
	s_load_b32 s8, s[6:7], 0x198                               // 00000000181C: F4000203 F8000198
	v_add_f32_e32 v0, s9, v0                                   // 000000001824: 06000009
	s_clause 0x1                                               // 000000001828: BF850001
	s_load_b32 s9, s[6:7], 0x1b0                               // 00000000182C: F4000243 F80001B0
	s_load_b32 s6, s[6:7], 0x1c8                               // 000000001834: F4000183 F80001C8
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000183C: BF870091
	v_add_f32_e32 v0, s10, v0                                  // 000000001840: 0600000A
	v_add_f32_e32 v0, s11, v0                                  // 000000001844: 0600000B
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001848: BF870091
	v_add_f32_e32 v0, s12, v0                                  // 00000000184C: 0600000C
	v_add_f32_e32 v0, s13, v0                                  // 000000001850: 0600000D
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001854: BF8700A1
	v_add_f32_e32 v0, s14, v0                                  // 000000001858: 0600000E
	s_waitcnt lgkmcnt(0)                                       // 00000000185C: BF89FC07
	v_add_f32_e32 v0, s3, v0                                   // 000000001860: 06000003
	s_ashr_i32 s3, s2, 31                                      // 000000001864: 86039F02
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001868: BF870099
	s_lshl_b64 s[2:3], s[2:3], 2                               // 00000000186C: 84828202
	v_add_f32_e32 v0, s8, v0                                   // 000000001870: 06000008
	s_add_u32 s2, s4, s2                                       // 000000001874: 80020204
	s_addc_u32 s3, s5, s3                                      // 000000001878: 82030305
	s_add_u32 s0, s2, s0                                       // 00000000187C: 80000002
	s_addc_u32 s1, s3, s1                                      // 000000001880: 82010103
	v_add_f32_e32 v0, s9, v0                                   // 000000001884: 06000009
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001888: BF870091
	v_add_f32_e32 v0, s6, v0                                   // 00000000188C: 06000006
	v_mul_f32_e32 v0, 0x3d4ccccd, v0                           // 000000001890: 100000FF 3D4CCCCD
	global_store_b32 v1, v0, s[0:1]                            // 000000001898: DC6A0000 00000001
	s_nop 0                                                    // 0000000018A0: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 0000000018A4: BFB60003
	s_endpgm                                                   // 0000000018A8: BFB00000
